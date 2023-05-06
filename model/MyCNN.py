import torch
from torch import nn as nn
from torch.nn import functional as F
import numpy as np
import argparse
import math
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
import os

parser=argparse.ArgumentParser()
parser.add_argument('--num_feats',type=int,default=8)
parser.add_argument('--kernel_size',type=int,default=3)
parser.add_argument('--picture_size',type=int,default=122)
parser.add_argument('--lr',type=float,default=1e-7)
parser.add_argument('--epoches',type=int,default=300)
parser.add_argument('--train_ratio',type=float,default=0.6)
parser.add_argument('--valid_ratio',type=float,default=0.2)
args=parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES']='1'

class ConvolutionNet(nn.Module):
    def __init__(self,args) -> None:
        super().__init__()
        self.args=args
        self.conv=nn.Conv2d(args.num_feats,out_channels=args.num_feats,kernel_size=args.kernel_size,)
        self.conv_size=args.picture_size-args.kernel_size+1
        self.maxpool=nn.MaxPool2d(args.kernel_size,stride=1)
        self.pool_size=self.conv_size-args.kernel_size+1
        self.clf=nn.Linear(self.pool_size*self.pool_size*args.num_feats,2)
        
    def forward(self,x):
        x1=self.conv(x)
        
        x1=self.maxpool(x1)
        x1=x1.flatten(1)
        x1=self.clf(x1)
        x1=F.relu(x1)
        x1=F.softmax(x1)
        return x1
        

def train():
    feats=np.load('/home/m21_huangzijun/pythonprojs/sichuan/Data2/pictures/pictures.npy',allow_pickle=True)
    feats=np.nan_to_num(feats)
    
    feats=torch.from_numpy(feats).type(torch.float32).cuda()
    feats=torch.permute(feats,(0,3,1,2))

    labels=np.load('/home/m21_huangzijun/pythonprojs/sichuan/Data2/pictures/labels.npy',allow_pickle=True)
    labels=torch.from_numpy(labels).type(torch.long).cuda()



    convo=ConvolutionNet(args).cuda()

    loss=nn.CrossEntropyLoss()
    optimizer=torch.optim.Adam(convo.parameters(),lr=args.lr)


    

    trainsize=math.floor(len(feats)*args.train_ratio)
    validsize=math.floor(len(feats)*(args.train_ratio+args.valid_ratio))

    # 验证集指标
    best_recall = 0
    best_f1 = 0
    best_auc = 0

    # 测试集指标
    test_acc = 0
    test_prec = 0
    test_recall = 0
    test_f1 = 0
    test_auc = 0

    xlabel=[]
    losses=[]
    torch.cuda.empty_cache()
    for epoch in range(args.epoches):
        convo.train()
        pre=convo(feats[:trainsize])
        optimizer.zero_grad()
        loss_v=loss(pre[:trainsize],labels[:trainsize])
        loss_v.backward()
        optimizer.step()
        xlabel.append(epoch)
        losses.append(loss_v.item())

        convo.eval()
        pre=convo(feats[trainsize:validsize])
        valid_pre = pre
        valid_label = labels[trainsize:validsize]
        valid_pre = valid_pre.detach().cpu().numpy()
        valid_label = valid_label.detach().cpu().numpy()
        valid_pre_y = np.argmax(valid_pre, axis=-1)
        pos_score = valid_pre[:, 1]
        acc = accuracy_score(valid_label, valid_pre_y)
        prec = precision_score(valid_label, valid_pre_y, average='macro')
        recall = recall_score(valid_label, valid_pre_y, average='macro')
        f1 = f1_score(valid_label, valid_pre_y, average='macro')
        auc = roc_auc_score(valid_label, pos_score)

        if True:
            best_auc = auc
            best_recall = recall
            best_f1 = f1
            
            pre=convo(feats[validsize:])
            test_pre = pre
            
            test_label = labels[validsize:]
            test_pre = test_pre.detach().cpu().numpy()
            
            test_label = test_label.detach().cpu().numpy()
            test_pre_y = np.argmax(test_pre, axis=-1)
            pos_score = test_pre[:, 1]
            acc = accuracy_score(test_label, test_pre_y,)
            prec = precision_score(
                test_label, test_pre_y, average='macro')
            recall = recall_score(
                test_label, test_pre_y, average='macro')
            f1 = f1_score(test_label, test_pre_y, average='macro')
            auc = roc_auc_score(test_label, pos_score)
            test_acc = acc
            test_prec = prec
            test_recall = recall
            test_f1 = f1
            test_auc = auc

    print('acc:{} prec:{} recall:{} f1:{} auc:{}'.format(
        test_acc, test_prec, test_recall, test_f1, test_auc))
    plt.plot(xlabel,losses)
    plt.savefig('loss_picture/loss.png')
    pass

if __name__=='__main__':
    train()