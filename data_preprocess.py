# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 19:53:43 2022

@author: Administrator
"""

"""weishenme_develop"""
"""weishenme_new_d"""
"""develop1"""
"""develop2"""
"""new_d1"""

import pandas as pd
import numpy as np
from collections import Counter
from matplotlib import pyplot as plt
import dgl 
import torch
import networkx as nx
from collections import Counter
from scipy.stats import entropy

def maxmin_len_funciton(data_call):
    list_user = list(data_call['phone_no_m'])
    score = Counter(list_user)
    list_keys = list(score.keys())
    max_len = 0
    min_len = score[list_user[0]]
    for i in list_keys:
        temp = score[i]
        if temp > max_len:
            max_len = temp
        if temp < min_len:
            min_len = temp
    return max_len, min_len

def graph_user2item(dict_month2df, month_index):
    dict_user2item = dict()
    for i in month_index:
        df_temp = dict_month2df[str(i)]
        user = list(df_temp['phone_no_m'])
        item = list(df_temp['opposite_no_m'])
        dict_user2item[('user', str(i), 'item')]=(user, item)
    hetero_user2item = dgl.heterograph(dict_user2item)
    return hetero_user2item

def dict_user2item(df_temp):
    list_user = list(df_temp['phone_no_m'])
    list_item = list(df_temp['opposite_no_m'])
    user = sorted(list(set(list_user)))
    dict_temp = dict()
    for i in user:
        dict_temp[i] = []
    for i in range(len(list_user)):
        dict_temp[list_user[i]].append(list_item[i])
    result = dict()
    for i in range(len(user)):
        result[user[i]] = set(dict_temp[user[i]])
    return result
    
def source2target(dict_u2i):
    user = sorted(list(dict_u2i.keys()))   # 进行排序
    num = len(user)
    source = list(range(num))   #加入自循环
    target = list(range(num))   #加入自循环
    for i in range(len(user)):
        for j in range(i+1, len(user)):
            if len(dict_u2i[user[i]] & dict_u2i[user[j]]) >= 1:
                source.append(user[i])
                target.append(user[j])
    return source, target

def graph_user2user(dict_month2df, month_index):
    dict_user2user = dict()
    user_unique = set()
    for i in month_index:
        print(i)
        df_temp = dict_month2df[str(i)]
        dict_u2i = dict_user2item(df_temp)
        user_source, user_target = source2target(dict_u2i)
        user_temp = set(user_source) | set(user_target)
        dict_user2user['user', str(i), 'user']=(user_source, user_target)
        user_unique = user_unique | user_temp
    hetero_user2user = dgl.heterograph(dict_user2user)
    return hetero_user2user

def normal_col(np_featrue):
    mx = np_featrue
    shape=mx.shape
    mx=mx.reshape((-1,shape[-1]))
    for k in range(mx.shape[-1]):
        mx[:,k]=(0.1+mx[:,k]-np.mean(mx[:,k]))/(0.1+np.std(mx[:,k]))
    mx=mx.reshape(shape)
    return mx 

def featureExtractor1(df_user_temp):
    # 通话类型
    int_calltype = df_user_temp['calltype_id']
    feat_calltype = [0, 0, 0]
    feat_calltype[int_calltype-1] = 1
    
    #通话时长
    int_calldur = df_user_temp['call_dur']
    feat_calldur = [np.sqrt(int_calldur), 0, 0, int_calldur, int_calldur, int_calldur, int_calldur]
    
    #对象重叠率
    feat_calltarget = [1, 1, 1, 0]
    
    #通话时间间隔
    int_calldur = df_user_temp['call_dur']
    feat_delta = [0, 0, int_calldur, int_calldur, int_calldur]
    
    #通话时刻分布
    np_hourday = np.array((df_user_temp['hour_day']))
    feat_calltime0 = np.sum((np_hourday >= 6) * (np_hourday <12))
    feat_calltime1 = np.sum((np_hourday >=12) * (np_hourday <18))
    feat_calltime2 = np.sum((np_hourday > 18) * (np_hourday <=23))
    feat_calltime = [feat_calltime0, feat_calltime1, feat_calltime2]
    feat_calltime = feat_calltime + list(np.array([feat_calltime0, feat_calltime1, feat_calltime2]))  
   
    feat_final = feat_calltype + feat_calldur + feat_calltarget + feat_delta + feat_calltime
    return feat_final

def featureExtractor(df_user_temp):
    
    # 通话类型
    list_calltype = list(df_user_temp['calltype_id'])
    num_total = len(list_calltype)
    calltype_counter = Counter(list_calltype)
    feat_calltype0 = calltype_counter[1]/num_total      #0 表示主叫
    feat_calltype1 = calltype_counter[2]/num_total      #1 表示被叫
    feat_calltype2 = calltype_counter[3]/num_total      #2 表示呼叫转移
    feat_calltype = [feat_calltype0, feat_calltype1, feat_calltype2]
    
    # 通话时长
    list_calldur = list(df_user_temp['call_dur'])
    feat_calldur0 = np.sqrt(np.mean(list_calldur))
    feat_calldur1 = np.std(list_calldur)
    feat_calldur2 = feat_calldur1/np.mean(list_calldur)
    feat_calldur3 = np.max(list_calldur)
    feat_calldur4 = np.min(list_calldur)
    feat_calldur5 = np.median(list_calldur)
    feat_calldur6 = np.sum(list_calldur)
    feat_calldur = [feat_calldur0, feat_calldur1, feat_calldur2, feat_calldur3, feat_calldur4, feat_calldur5, feat_calldur6]
    
    # 通话对象重叠率
    list_receiver = list(df_user_temp['opposite_no_m'])
    num_receiver = np.sum(len(set(list_receiver)))
    feat_calltarget0 = num_receiver
    feat_calltarget1 = len(list_receiver)
    feat_calltarget2 = num_receiver/feat_calltarget1 
    calltarget_counter = Counter(list_receiver)
    distribution_calltarget = np.array(list(calltarget_counter.values()))/len(list_receiver)
    feat_calltarget3 = entropy(distribution_calltarget)
    feat_calltarget = [feat_calltarget0, feat_calltarget1, feat_calltarget2, feat_calltarget3]
    
    #通话时间间隔
    list_delta = list(df_user_temp['TimeDelta']) 
    list_delta = list_delta[1:]
    feat_delta0 = np.sqrt(np.mean(list_delta))
    feat_delta1 = np.std(list_delta)
    feat_delta2 = np.max(list_delta)
    feat_delta3 = np.min(list_delta)
    feat_delta4 = np.median(list_delta)
    
#        feat_delta2 = feat_delta1/np.mean(list_delta)
    feat_delta = [feat_delta0, feat_delta1, feat_delta2, feat_delta3, feat_delta4]
        
    #通话时刻分布:（hour of day）、（weekday or weekend）
    np_hourday = np.array(list(df_user_temp['hour_day']))
    feat_calltime0 = np.sum((np_hourday >= 6) * (np_hourday <12))
    feat_calltime1 = np.sum((np_hourday >=12) * (np_hourday <18))
    feat_calltime2 = np.sum((np_hourday > 18) * (np_hourday <=23))
    feat_calltime = [feat_calltime0, feat_calltime1, feat_calltime2]
    feat_calltime = feat_calltime + list(np.array([feat_calltime0, feat_calltime1, feat_calltime2])/num_total)  
    
    feat_final = feat_calltype + feat_calldur + feat_calltarget + feat_delta + feat_calltime
    return feat_final


# 程序入口
data_user_path = r'./Data/data_user.csv'
data_call_path = r'./Data/data_voc1.csv'

data_user = pd.read_csv(data_user_path)
data_call = pd.read_csv(data_call_path)

# 对用户及其实践戳进行排序
data_call = data_call.sort_values(by=['phone_no_m','time_stamp'])

# 加入月份和周的概念
stand_time = pd.to_datetime(data_call['start_datetime'])
temp_month = list(stand_time.dt.month)
name_week = list(stand_time.dt.week)
name_year = list(stand_time.dt.year)
name_month = [name_year[i]*100+temp_month[i] for i in range(len(temp_month))]
data_call['name_month'] = name_month
data_call['name_week'] = name_week

# 按照月进行时间划分
month_index = sorted(list(set(data_call['name_month'])))
data_call_monthIndex = data_call.set_index('name_month')
dict_month2df = {}
for i in month_index:
    dict_month2df[str(i)]=data_call_monthIndex.loc[i]

print(dict_month2df.keys())    

# 提取用户-对象的二部图数据
hetero_user2item = graph_user2item(dict_month2df, month_index)
# 提取用户-用户的同质图
hetero_user2user = graph_user2user(dict_month2df, month_index)
list_etype = sorted(hetero_user2user.etypes)
list_graph_u2u = [dgl.to_bidirected(hetero_user2user[i]) for i in list_etype]

df_user = data_user.sort_values(by=['user'])
df_user1 = df_user.set_index('label')
df_normal = df_user1.loc[0]
df_fraud = df_user1.loc[1]

def basicfeatureMaker(data_user):
    df_user = data_user.sort_values(by=['user'])
    df_user_temp = data_user.set_index('user')
    user_list = sorted(list(df_user['user']))
    np_userfeature = np.zeros((len(user_list), 5))
    for i in range(len(user_list)):
        se_temp = df_user_temp.loc[i]
        idcard_cnt = list(se_temp[0])
        arpu = list(se_temp[1:-1])
        
        feature0 = list(np.array([np.float(idcard_cnt == 0), np.float(idcard_cnt == 1), np.float(idcard_cnt >1)]))
        feature1 = arpu
        feature2 = [np.sum(arpu), np.mean(arpu), np.std(arpu), np.max(arpu), np.min(arpu), np.median(arpu)]
        feature3 = 
        
        
    

def callfeatureMaker(hetero_user2user, dict_month2df, data_user):
    # 对用户进行排序
    df_user = data_user.sort_values(by=['user'])
    df_user_temp = data_user.set_index('user')
    num_user = hetero_user2user.num_nodes()
    label = torch.LongTensor(df_user['label'])
    hetero_user2user.ndata['label'] = label
    list_keys = sorted(dict_month2df.keys())
    
    for i in range(len(list_keys)):
        print(list_keys[i])
        df_temp = dict_month2df[list_keys[i]]
        df_temp = df_temp.sort_values(by=['phone_no_m', 'time_stamp'])
        users = sorted(list(set(df_temp['phone_no_m'])))
        df_temp1 = df_temp.set_index('phone_no_m')
        np_feature0 = np.zeros((num_user, 25))
        np_feature1 = np.zeros((num_user, 25))
        np_feature2 = np.zeros((len(users), 25))
        for j in range(len(users)):
            df_user_temp = df_temp1.loc[users[j]]
            if type(df_user_temp) == pd.pandas.core.series.Series:
                feature_temp = featureExtractor1(df_user_temp)
            else:
                feature_temp = featureExtractor(df_user_temp)
            
            np_feature1[users[j]] = feature_temp
            np_feature2[j] = feature_temp
        
        np_feature1 = normal_col(np_feature1)           #对所有节点进行归一化
        np_feature2 = normal_col(np_feature2)           #对有电话记录的用户进行归一化
        for k in range(len(users)):
            np_feature0[users[k]] = np_feature2[k]

        if np.sum(np.isnan(np_feature1)) == 0:
            print('np_feature1：without nan')
        else:
            print('np_feature1：nan exists')
            
        if np.sum(np.isnan(np_feature2)) == 0:
            print('np_feature2: without nan')
        else:
            print('np_feature2：nan exists')
        

        
        hetero_user2user[list_keys[i]].ndata['feature0'] = torch.tensor(np_feature0)
        hetero_user2user[list_keys[i]].ndata['feature1'] = torch.tensor(np_feature1)
        

def time_encode(x):
    if type(x) == float:
        x = torch.tensor(x).unsqueeze(0)
    x *= 100
    x = x.unsqueeze(1)
    pe = torch.zeros((len(x), 8))
    coef = torch.exp((torch.arange(0, 8, 2, dtype=torch.float) * -(np.log(10000.0) / 8)))
    pe[:,0::2] = torch.sin(x * coef)
    pe[:,1::2] = torch.cos(x * coef)
    return pe

#data_call1 = data_call.set_index('phone_no_m')
#list_user = sorted(list(set((data_call1.index))))
#max_len, min_len = maxmin_len_funciton(data_call) 
#dict_user2call = {}
#for i in list_user:
#    pd_temp = data_call1.loc[i]
#    list_opposite = list(pd_temp['opposite_no_m'])
#    list_calltype = list(pd_temp['calltype_id'])
#    list_calltime = list(pd_temp['time_stamp'])
#    list_timedelta= list(pd_temp['TimeDelta'])
#    list_label = list(set(pd_temp['label']))
#    list_all = [list_opposite, list_calltype, list_calltime, list_timedelta, list_label]
#    dict_user2call[i] = list_all

