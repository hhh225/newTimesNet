import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
from layers.Embed import DataEmbedding
from layers.Conv_Blocks import Inception_Block_V1
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

import argparse
parser = argparse.ArgumentParser()
# basic config
parser.add_argument('--task_name', type=str,  default='classification',
                    help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
parser.add_argument('--is_training', type=int,  default=1, help='status')
parser.add_argument('--model_id', type=str,
                    default='EthanolConcentration', help='model id')
parser.add_argument('--model', type=str,  default='TimesNet',
                    help='model name, options: [Autoformer, Transformer, TimesNet]')
parser.add_argument('--lr', type=float, default=1e-7)

# forecasting task
parser.add_argument('--seq_len', type=int, default=96,
                    help='input sequence length')
parser.add_argument('--label_len', type=int, default=48,
                    help='start token length')
parser.add_argument('--pred_len', type=int, default=96,
                    help='prediction sequence length')
parser.add_argument('--seasonal_patterns', type=str,
                    default='Monthly', help='subset for M4')

# inputation task
parser.add_argument('--mask_rate', type=float, default=0.25, help='mask ratio')

# anomaly detection task
parser.add_argument('--anomaly_ratio', type=float,
                    default=0.25, help='prior anomaly ratio (%)')

# model define
parser.add_argument('--top_k', type=int, default=3, help='for TimesBlock')
parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')
parser.add_argument('--enc_in', type=int, default=32, help='encoder input size')
parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
parser.add_argument('--c_out', type=int, default=7, help='output size')
parser.add_argument('--d_model', type=int, default=32,
                    help='dimension of model')
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
parser.add_argument('--e_layers', type=int, default=3,
                    help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=1,
                    help='num of decoder layers')
parser.add_argument('--d_ff', type=int, default=32, help='dimension of fcn')
parser.add_argument('--moving_avg', type=int, default=25,
                    help='window size of moving average')
parser.add_argument('--factor', type=int, default=1, help='attn factor')
parser.add_argument('--distil', action='store_false',
                    help='whether to use distilling in encoder, using this argument means not using distilling',
                    default=True)
parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
parser.add_argument('--embed', type=str, default='timeF',
                    help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--activation', type=str,
                    default='gelu', help='activation')
parser.add_argument('--output_attention', action='store_true',
                    help='whether to output attention in ecoder')

parser.add_argument('--freq', type=str, default='h',
                    help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
parser.add_argument('--epoches', type=int, default=20)
# optimization
parser.add_argument('--num_workers', type=int, default=10,
                    help='data loader num workers')
parser.add_argument('--train_epochs', type=int,
                    default=30, help='train epochs')
parser.add_argument('--batch_size', type=int, default=256,
                    help='batch size of train input data')
parser.add_argument('--patience', type=int, default=10,
                    help='early stopping patience')
parser.add_argument('--learning_rate', type=float,
                    default=0.001, help='optimizer learning rate')
parser.add_argument('--des', type=str, default='Exp', help='exp description')
parser.add_argument('--loss', type=str, default='MSE', help='loss function')
parser.add_argument('--lradj', type=str, default='type1',
                    help='adjust learning rate')
parser.add_argument('--use_amp', action='store_true',
                    help='use automatic mixed precision training', default=False)
parser.add_argument('--shuffle', default=True)

# GPU
parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', action='store_true',
                    help='use multiple gpus', default=False)
parser.add_argument('--devices', type=str, default='0,1,2,3',
                    help='device ids of multile gpus')


args, _ = parser.parse_known_args()


def adjust_learning_rate(optimizer, epoch, args):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.lr * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))


class MyDataset(Dataset):
    def __init__(self, data: np.ndarray, label: np.ndarray) -> None:
        super().__init__()
        self.data = data
        self.label = label

    def __getitem__(self, index):
        return self.data[index], self.label[index]

    def __len__(self):
        return len(self.data)


def FFT_for_Period(x, k=2):   # B,T,C
    # [B, T, C]
    xf = torch.fft.rfft(x, dim=1)
    # find period by amplitudes
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list
    return period, abs(xf).mean(-1)[:, top_list]


class TimesBlock(nn.Module):
    def __init__(self, configs):
        # 所需的参数, seq_len, pred_len, top_k, d_model, d_ff, num_kernels,
        super(TimesBlock, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.k = configs.top_k
        # parameter-efficient design
        self.conv = nn.Sequential(
            Inception_Block_V1(configs.d_model, configs.d_ff,
                               num_kernels=configs.num_kernels),
            nn.GELU(),
            Inception_Block_V1(configs.d_ff, configs.d_model,
                               num_kernels=configs.num_kernels)
        )

    def forward(self, x): # x (B,T,N)
        B, T, N = x.size()
        period_list, period_weight = FFT_for_Period(x, self.k)

        res = []
        for i in range(self.k):
            period = period_list[i]
            # padding
            if (self.seq_len + self.pred_len) % period != 0:
                length = (
                    ((self.seq_len + self.pred_len) // period) + 1) * period
                padding = torch.zeros(
                    [x.shape[0], (length - (self.seq_len + self.pred_len)), x.shape[2]]).to(x.device)
                out = torch.cat([x, padding], dim=1)
            else:
                length = (self.seq_len + self.pred_len)
                out = x
            # reshape
            out = out.reshape(B, length // period, period,
                              N).permute(0, 3, 1, 2).contiguous()
            # 2D conv: from 1d Variation to 2d Variation
            out = self.conv(out)
            # reshape back
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
            res.append(out[:, :(self.seq_len + self.pred_len), :])
        res = torch.stack(res, dim=-1)
        # adaptive aggregation
        period_weight = F.softmax(period_weight, dim=1)
        period_weight = period_weight.unsqueeze(
            1).unsqueeze(1).repeat(1, T, N, 1)
        res = torch.sum(res * period_weight, -1)
        # residual connection
        res = res + x
        return res


class Model(nn.Module):
    """
    Paper link: https://openreview.net/pdf?id=ju_Uqw384Oq
    """

    # 所需参数 config task_name, label_len, e_layers, enc_in, embed, freq, dropout, c_out, num_class
    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.model = nn.ModuleList([TimesBlock(configs)
                                    for _ in range(configs.e_layers)])
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)
        self.layer = configs.e_layers
        self.layer_norm = nn.LayerNorm(configs.d_model)
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.predict_linear = nn.Linear(
                self.seq_len, self.pred_len + self.seq_len)
            self.projection = nn.Linear(
                configs.d_model, configs.c_out, bias=True)
        if self.task_name == 'imputation' or self.task_name == 'anomaly_detection':
            self.projection = nn.Linear(
                configs.d_model, configs.c_out, bias=True)
        if self.task_name == 'classification':
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(
                configs.d_model * configs.seq_len, configs.num_class)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        # embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # [B,T,C]
        enc_out = self.predict_linear(enc_out.permute(0, 2, 1)).permute(
            0, 2, 1)  # align temporal dimension
        # TimesNet
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))
        # porject back
        dec_out = self.projection(enc_out)

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * \
            (stdev[:, 0, :].unsqueeze(1).repeat(
                1, self.pred_len + self.seq_len, 1))
        dec_out = dec_out + \
            (means[:, 0, :].unsqueeze(1).repeat(
                1, self.pred_len + self.seq_len, 1))
        return dec_out

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        # Normalization from Non-stationary Transformer
        means = torch.sum(x_enc, dim=1) / torch.sum(mask == 1, dim=1)
        means = means.unsqueeze(1).detach()
        x_enc = x_enc - means
        x_enc = x_enc.masked_fill(mask == 0, 0)
        stdev = torch.sqrt(torch.sum(x_enc * x_enc, dim=1) /
                           torch.sum(mask == 1, dim=1) + 1e-5)
        stdev = stdev.unsqueeze(1).detach()
        x_enc /= stdev

        # embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # [B,T,C]
        # TimesNet
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))
        # porject back
        dec_out = self.projection(enc_out)

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * \
            (stdev[:, 0, :].unsqueeze(1).repeat(
                1, self.pred_len + self.seq_len, 1))
        dec_out = dec_out + \
            (means[:, 0, :].unsqueeze(1).repeat(
                1, self.pred_len + self.seq_len, 1))
        return dec_out

    def anomaly_detection(self, x_enc):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        # embedding
        enc_out = self.enc_embedding(x_enc, None)  # [B,T,C]
        # TimesNet
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))
        # porject back
        dec_out = self.projection(enc_out)

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * \
            (stdev[:, 0, :].unsqueeze(1).repeat(
                1, self.pred_len + self.seq_len, 1))
        dec_out = dec_out + \
            (means[:, 0, :].unsqueeze(1).repeat(
                1, self.pred_len + self.seq_len, 1))
        return dec_out

    def classification(self, x_enc, x_mark_enc):
        # embedding
        enc_out = self.enc_embedding(x_enc, None)  # [B,T,C] 
        # TimesNet
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))

        # Output
        # the output transformer encoder/decoder embeddings don't include non-linearity
        output = self.act(enc_out)
        output = self.dropout(output)
        # zero-out padding embeddings
        output = output * x_mark_enc.unsqueeze(-1)
        # (batch_size, seq_length * d_model)
        output = output.reshape(output.shape[0], -1)
        output = self.projection(output)  # (batch_size, num_classes)
        return output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        if self.task_name == 'imputation':
            dec_out = self.imputation(
                x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
            return dec_out  # [B, L, D]
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            return dec_out  # [B, L, D]
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc, x_mark_enc)
            dec_out=F.softmax(dec_out,dim=-1)
            return dec_out  # [B, N]
        return None


def padding_mask(lengths, max_len=None):
    """
    Used to mask padded positions: creates a (batch_size, max_len) boolean mask from a tensor of sequence lengths,
    where 1 means keep element at this position (time step)
    """
    batch_size = lengths.numel()
    # trick works because of overloading of 'or' operator for non-boolean types
    max_len = max_len
    return (torch.arange(0, max_len, device=lengths.device)
            .type_as(lengths)
            .repeat(batch_size, 1)
            .lt(lengths.unsqueeze(1)))


def collan_fn(x, max_len=None):
    batch_size = len(x)
    features, labels = zip(*x)

    lengths = [X.shape[0] for X in features]
    if max_len == None:
        max_len = max(lengths)

    X = np.zeros((batch_size, max_len, features[0].shape[-1]))
    
    for i in range(batch_size):
        end = min(lengths[i], max_len)
        X[i, :end, :] = features[i][:end, :]

    targets = torch.tensor(labels)
    paded_mask = padding_mask(torch.tensor(lengths,dtype=torch.int16), max_len=max_len)


    X=torch.from_numpy(X)

    return X, targets, paded_mask


def train():

    oneday=np.load('/home/m21_huangzijun/pythonprojs/sichuan/data_after/timefreq/x_D.npy',allow_pickle=True)
    label=np.load('/home/m21_huangzijun/pythonprojs/sichuan/data_after/timefreq/y.npy',allow_pickle=True)

    oneday=np.nan_to_num(oneday)

    train_x,test_x,train_y,test_y=train_test_split(oneday,label,test_size=0.2,random_state=5)



    

    one_week = np.load(
        '/home/m21_huangzijun/pythonprojs/sichuan/data_after/mymodel5_2/x_train_1.npy')
    one_week_t = np.load(
        '/home/m21_huangzijun/pythonprojs/sichuan/data_after/mymodel5_2/x_test_1.npy')
    
    one_week=np.nan_to_num(one_week)
    one_week_t=np.nan_to_num(one_week_t)

    label = np.load(
        f'/home/m21_huangzijun/pythonprojs/sichuan/data_after/mymodel5_2/y_train.npy')
    label_t = np.load(
        '/home/m21_huangzijun/pythonprojs/sichuan/data_after/mymodel5_2/y_test.npy')

    lengths = [len(x) for x in oneday]
    args.seq_len = max(lengths)

    dataset = MyDataset(train_x, train_y)

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=args.shuffle,
                            collate_fn=lambda x: collan_fn(x, max_len=args.seq_len))

    if args.task_name == 'classification':  # 重新设置 seq_len, pred_len, enc_in, num_class
        args.pred_len = 0
        args.num_class = 2
        args.enc_in = one_week.shape[-1]
    if args.use_gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(
            args.gpu) if not args.use_multi_gpu else args.devices
        device = torch.device('cuda:{}'.format(args.gpu))
    else:
        device = torch.device('cpu')

    model = Model(args).float()
    model=model.to(device)
    if args.use_multi_gpu:
        model = nn.DataParallel(model, device_ids=args.device_ids)

    criterion = nn.CrossEntropyLoss()
    model_optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    trainlosses=list()
    xlabel=list()

    for epoch in range(args.epoches):
        trainloss = list()
        for i, (batch_x, label, padding_mask) in enumerate(dataloader):
            model_optim.zero_grad()
            batch_x = batch_x.float().to(device)
            padding_mask = padding_mask.float().to(device)
            label = label.to(device)
            output = model(batch_x, padding_mask, None, None)
            loss = criterion(output, label.long().squeeze(-1))
            trainloss.append(loss.item())
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=4)
            model_optim.step()
        xlabel.append(epoch)
        trainloss = np.average(trainloss)
        trainlosses.append(trainloss)
        if (epoch+1) % 5 == 0:
            adjust_learning_rate(model_optim, epoch+1, args)

    # one_week_t=torch.from_numpy(one_week_t).type(torch.float32).to(device)

    plt.cla()
    plt.plot(xlabel,trainlosses)
    plt.savefig('loss.png')
    

    
    test_dataset=MyDataset(test_x,test_y)
    test_dataloader=DataLoader(test_dataset,batch_size=len(test_x),collate_fn=lambda x:collan_fn(x,max_len=args.seq_len))

    for i,(batch_x,label,paded_mask) in enumerate(test_dataloader):

        batch_x = batch_x.float().to(device)
        paded_mask = paded_mask.float().to(device)
        test_output=model(batch_x,paded_mask,None,None)
        
    test_output=test_output.detach().cpu().numpy()
    test_output_y = np.argmax(test_output, axis=-1)
    pos_score = test_output[:, 1]

    acc = accuracy_score(test_y, test_output_y,)
    prec = precision_score(test_y, test_output_y,average='macro')
    rec = recall_score(test_y, test_output_y,average='macro')
    f1 = f1_score(test_y, test_output_y,average='macro')
    auc=roc_auc_score(test_y,pos_score,average='macro')

    print('acc:{} prec:{} rec:{} f1:{} auc:{}'.format(acc,prec,rec,f1,auc))

    



if __name__ == '__main__':
    train()
