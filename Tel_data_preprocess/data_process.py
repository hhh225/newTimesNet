# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 16:15:15 2022

@author: Administrator
"""

import time
import pandas as pd
import numpy as np
from collections import Counter
import os
MIN_THRESHOLD = 20

def time_delata(list_target_user, list_target_timestamp):
    
    list1 = list(list_target_timestamp)
    list2 = [list1[0]]+list1[:-1]
    delta = list(np.array(list1) - np.array(list2))
    list_index = list_target_user
    point = []
    for i in range(len(list_index)-1):
        if list_index[i] != list_index[i+1]:
            point.append(i+1)
    for j in point:
        delta[j] = 0 
    return delta

# 删除通话数量少于num_threshold的用户及其通话记录
def Remove_user(data_voc, num_threshold):
    list_caller = list(data_voc['phone_no_m'])
    user_counter = Counter(list_caller)
    key_caller = list(user_counter.keys())
    list_removeuser = []
    for i in key_caller:
        if user_counter[i] < num_threshold:
            list_removeuser.append(i)
    pd_temp = data_voc.set_index('phone_no_m')
    pd_temp1 = pd_temp.drop(index=list_removeuser)
    pd_temp2 = pd_temp1.reset_index(drop=False)
    return pd_temp2

print('hello world')

data_path = os.environ['HOME']+r'/pythonprojs/sichuan/data/0527/train'
data_name = ['/train_app.csv', '/train_sms.csv', '/train_user.csv', '/train_voc.csv']
path_voc = data_path + data_name[3]
path_user = data_path + data_name[2]

data_voc_temp = pd.read_csv(path_voc)
data_user = pd.read_csv(path_user)
data_voc = pd.DataFrame()

print(data_voc_temp.columns)

#缺失值所在行进行删除
data_voc['phone_no_m'] = data_voc_temp['phone_no_m']
data_voc['opposite_no_m'] = data_voc_temp['opposite_no_m']
data_voc['calltype_id'] = data_voc_temp['calltype_id']
data_voc['start_datetime'] = data_voc_temp['start_datetime']
data_voc['call_dur'] = data_voc_temp['call_dur']
data_voc = data_voc.dropna()

#删除通话记录少于X条的用户
num_threshold = MIN_THRESHOLD
data_voc = Remove_user(data_voc, num_threshold)

# 求caller和receiver之间的交集
set_caller = set(data_voc['phone_no_m'])
set_receiver = set(data_voc['opposite_no_m'])
set_union = set_caller & set_receiver

# 对caller和receiver分别进行编码
index2caller = sorted(list(set(data_voc['phone_no_m'])))
caller2index = dict()
for idx, value in enumerate(index2caller):
    caller2index[value] = idx
    
index2receiver = sorted(list(set(data_voc['opposite_no_m'])))
reveiver2index = dict()
for idx, value in enumerate(index2receiver):
    reveiver2index[value] = idx
    
temp1 = [caller2index[i] for i in data_voc['phone_no_m']]
temp2 = [reveiver2index[i] for i in data_voc['opposite_no_m']]


################caller和receiver一起进行编码####
index2user=list(set(data_voc['phone_no_m'])|set(data_voc['opposite_no_m']))
user2index=dict()
count=0
for idx,value in enumerate(index2user):
    user2index[value]=count
    count+=1
temp1=[user2index[i] for i in data_voc['phone_no_m']]
temp2=[user2index[i] for i in data_voc['opposite_no_m']]


# 将voc表中的人都换成编码
data_voc['phone_no_m'] = temp1 
data_voc['opposite_no_m'] = temp2

# 将voc中的时间换成时间戳 time_stamp
stand_time = pd.to_datetime(data_voc['start_datetime'])
data_voc['hour_day'] = stand_time.dt.hour
data_voc['day_week'] = stand_time.dt.dayofweek
data_voc['day_year'] = stand_time.dt.dayofyear
data_voc['time_stamp'] = stand_time.apply(lambda x: time.mktime(x.timetuple())) # 使用秒数来表示时间

# 将user表中的id都换成index, index保存在'user'列中
data_user1 = data_user
user_idx = []
user_code = list(data_user1['phone_no_m'])
for i in user_code:
    if i in list(user2index.keys()):
        temp = user2index[i]
        user_idx.append(temp)
    else:
        user_idx.append(9999)
data_user1['user'] = user_idx
data_user1 = data_user1.sort_values(by=['user'])
data_user2 = data_user1.set_index('user')
data_user2 = data_user2.drop([9999])
data_user2 = data_user2.reset_index()
del data_user2['city_name']
del data_user2['county_name']
data_user2 = data_user2.fillna(0)
list_user = list(data_user2['user'])
list_label =list(data_user2['label'])
user2label = dict(list(zip(list_user, list_label)))

#将data_voc中的用户和标签绑定
final_user = list(data_voc['phone_no_m'])
final_label= [user2label[i] for i in final_user]
data_voc['label'] = final_label


data_voc1 = data_voc.sort_values(by=['phone_no_m','time_stamp'])
list_target_user = list(data_voc1['phone_no_m'])
list_target_timestamp = list(data_voc1['time_stamp'])
TimeDelta = time_delata(list_target_user, list_target_timestamp)
data_voc1['TimeDelta'] = TimeDelta #与上一次通话的时间间隔

# data_voc1按人的编码和时间排序
# data_voc2完全按时间排序

data_voc1.to_csv(r'Data2/data_voc1.csv',sep=',',index=False, header=True)
data_voc2 = data_voc1.sort_values(by=['time_stamp']) #根据time stamp排序
data_voc2.to_csv(r'Data2/data_voc2.csv',sep=',',index=False, header=True)
data_user2.to_csv(r'Data2/data_user.csv',sep=',',index=False, header=True)