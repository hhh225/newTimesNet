import unittest
import os
import pandas as pd
import numpy as np
from collections import defaultdict
from contextlib import contextmanager
from time import time, sleep
import math


@contextmanager
def timespend(str1):
    t1 = time()
    yield
    t2 = time()
    print('{} spend time:{}'.format(str1, t2-t1))


def test1():
    feas = np.load('Data2/pictures/feas.npy')
    print(feas[0][:4])


def test2():
    df1 = pd.DataFrame({'t': ['2020-1-1 19:00', '2020-1-1 20:00',
                       '2020-1-2 19:00'], 'k2': [4, 4, 6], 'v': [1, 2, 4]})
    df1['t'] = pd.to_datetime(df1['t'])
    df1['sum'] = df1.groupby(pd.Grouper(key='t', freq='D'))[
        'v'].transform('sum')
    print(df1)


def test3():
    arr1 = np.array([0, 0, 2, 2, 4])
    print(np.where((arr1 > 1) & (arr1 < 3))[0])


def test4():
    arr1 = np.load('Data2/pictures/feas.npy', allow_pickle=True)
    print(arr1.shape)

def test5():
    pictures=np.load('/home/m21_huangzijun/pythonprojs/sichuan/Data2/pictures/pictures.npy')
    
    pass


def main():
    voc1 = pd.read_csv('Data2/data_voc1.csv')
    voc1['start_datetime'] = pd.to_datetime(voc1['start_datetime'])
    days = [i for _, i in voc1.groupby(
        by=pd.Grouper(key='start_datetime', freq='D'))]
    user = pd.read_csv('Data2/data_user.csv')
    user = user.set_index(keys='user', drop=False)
    list_user = list(user['user'])
    label_dict = dict(zip(user['user'], user['label']))
    feas = list()
    labels = list()
    user_feat = defaultdict(list)

    # 一起处理
    voc1['nuniqueContact'] = voc1.groupby(['phone_no_m', pd.Grouper(
        key='start_datetime', freq='D')])['opposite_no_m'].transform('nunique')
    callvoc = voc1.loc[voc1['calltype_id'] ==
                       1][['phone_no_m', 'start_datetime']]
    callvoc['calltimes'] = callvoc.groupby(['phone_no_m', pd.Grouper(
        key='start_datetime', freq='D')])['phone_no_m'].transform('count')
    receivevoc = voc1.loc[voc1['calltype_id']
                          == 2][['phone_no_m', 'start_datetime']]
    receivevoc['receivetimes'] = receivevoc.groupby(['phone_no_m', pd.Grouper(
        key='start_datetime', freq='D')])['phone_no_m'].transform('count')
    callvoc = callvoc[['phone_no_m', 'start_datetime',
                       'calltimes']].drop_duplicates()
    receivevoc = receivevoc[['phone_no_m',
                             'start_datetime', 'receivetimes']].drop_duplicates()
    voc1 = voc1.merge(callvoc, how='left', on=['phone_no_m', 'start_datetime'])
    voc1 = voc1.merge(receivevoc, how='left', on=[
                      'phone_no_m', 'start_datetime'])
    workvoc = voc1.loc[(voc1['hour_day'] >= 9) & (voc1['hour_day'] <= 18)]
    workvoc['worktime'] = workvoc.groupby(['phone_no_m', pd.Grouper(
        key='start_datetime', freq='D')])['phone_no_m'].transform('count')
    workvoc = workvoc[['phone_no_m', 'start_datetime',
                       'worktime']].drop_duplicates()
    voc1 = voc1.merge(workvoc, how='left', on=['phone_no_m', 'start_datetime'])
    napvoc = voc1.loc[~((voc1['hour_day'] >= 9) & (voc1['hour_day'] <= 18))]
    napvoc['naptime'] = napvoc.groupby(['phone_no_m', pd.Grouper(
        key='start_datetime', freq='D')])['phone_no_m'].transform('count')
    napvoc = napvoc[['phone_no_m', 'start_datetime',
                     'naptime']].drop_duplicates()
    voc1 = voc1.merge(napvoc, how='left', on=['phone_no_m', 'start_datetime'])
    voc1['calldur_mean'] = voc1.groupby(['phone_no_m', pd.Grouper(
        key='start_datetime', freq='D')])['call_dur'].transform('mean')
    voc1['calldur_var'] = voc1.groupby(['phone_no_m', pd.Grouper(
        key='start_datetime', freq='D')])['call_dur'].transform('var')
    voc1['delta_mean'] = voc1.groupby(['phone_no_m', pd.Grouper(
        key='start_datetime', freq='D')])['TimeDelta'].transform('mean')

    for _, day in voc1.groupby(pd.Grouper(key='start_datetime', freq='D')):  # 按天切片
        ps = day.groupby('phone_no_m')
        temp = set(list_user)
        for pid, p in ps:
            temp.remove(pid)
            nuqinueContact = p['nuniqueContact'].iloc[0]
            calltimes = p['calltimes'].iloc[0]
            receivetime = p['receivetimes'].iloc[0]

            worktime = p['worktime'].iloc[0]
            naptime = p['naptime'].iloc[0]
            calldur_mean = p['calldur_mean'].iloc[0]
            calldur_var = p['calldur_var'].iloc[0]
            delta_mean = p['delta_mean'].iloc[0]
            user_feat[pid].append([nuqinueContact, calltimes, receivetime,
                                   worktime, naptime, calldur_mean, calldur_var, delta_mean])
        for pid in list(temp):
            user_feat[pid].append([0, 0, 0, 0, 0, 0, 0, 0])
        # for pid in list_user:
        #     if pid not in ps: # 这天没打电话
        #         user_feat[pid].append([0, 0, 0, 0, 0, 0, 0, 0])
        #     else:
        #         p = ps.get_group(pid)
        #         nuqinueContact = p['nuniqueContact'].iloc[0]
        #         calltimes = p['calltimes'].iloc[0]
        #         receivetime = p['receivetime'].iloc[0]
        #         worktime = p['worktime'].iloc[0]
        #         naptime = p['naptime'].iloc[0]
        #         calldur_mean = p['calldur_mean'].iloc[0]
        #         calldur_var = p['calldur_var'].iloc[0]
        #         delta_mean = p['delta_mean'].iloc[0]
        #         user_feat[pid].append([nuqinueContact, calltimes, receivetime,
        #                               worktime, naptime, calldur_mean, calldur_var, delta_mean])

    # 循环处理
    '''
    with timespend('feature processing') as ts:
        day_count = 0
        for day in days:
            ps = [p for _, p in day.groupby('phone_no_m')]
            feature_count = 0
            temp = set(list_user.copy())
            p_count = 0
            for p in ps:

                uqinueContact = np.unique(
                    p['opposite_no_m'].to_numpy(dtype=np.int32))
                nuqinueContact = len(uqinueContact)
                feature_count += 1
                numpy_p = p['calltype_id'].to_numpy(dtype=np.int16)
                calltimes = len(np.where(numpy_p == 1)[0])
                receivetime = len(np.where(numpy_p == 2)[0])
                feature_count += 1
                feature_count += 1
                numpy_p = p['hour_day'].to_numpy(dtype=np.int16)
                worktime = len(np.where((numpy_p >= 9) & (numpy_p <= 18))[0])
                feature_count += 1
                alltime = len(numpy_p)
                naptime = alltime-worktime
                worktime /= alltime
                naptime /= alltime
                feature_count += 1
                calldur_mean = p['call_dur'].mean()
                feature_count += 1
                calldur_var = p['call_dur'].var()
                feature_count += 1
                timedelta = p['TimeDelta'].mean()
                feature_count += 1

                id = p['phone_no_m'].iloc[0]
                user_feat[id].append([nuqinueContact, calltimes, receivetime,
                                     worktime, naptime, calldur_mean, calldur_var, timedelta])
                temp.remove(id)

            for id in list(temp):
                user_feat[id].append([0 for i in range(feature_count)])
            print("day{}".format(day_count))
            day_count += 1
    '''
    with timespend('feature processing2') as ts:
        for pid in list_user:
            feas.append(user_feat[pid])
            labels.append(label_dict[pid])

    feas = np.array(feas)
    feas = np.nan_to_num(feas)
    list_user = np.array(list_user)
    labels = np.array(labels)
    np.save('Data2/pictures/ids.npy', list_user)
    np.save('Data2/pictures/feas.npy', feas)
    np.save('Data2/pictures/labels.npy', labels)

def relativePositionMatrix(matrix,k):
  N=len(matrix)
  ceil=math.ceil(N/k)
  floor=math.floor(N/k)
  m=ceil
  miu=np.mean(matrix,axis=0)
  std=np.std(matrix,axis=0)
  z=(matrix-miu)/std
  new_x=list()
  if ceil-floor==0:
    for i in range(m):
      tilde_x_i=0
      for j in range(k*(i-1)+1,k*i):
        tilde_x_i+=z[j]
      tilde_x_i/=k
      new_x.append(tilde_x_i)
  else:
    for i in range(m-1):
      tilde_x_i=np.zeros_like(z[0])
      for j in range(k*(i-1)+1,k*i):
        tilde_x_i+=z[j]
      new_x.append(tilde_x_i)
    tilde_x_i=np.zeros_like(z[0])
    for j in range(k*(m-1)+1,N):
      tilde_x_i+=z[j]
    tilde_x_i/=1/(N-k*(m-1))
    new_x.append(tilde_x_i)
  
  new_matrix=[[0 for j in range(len(new_x))] for i in range(len(new_x))]
  for i in range(len(new_x)):
    for j in range(len(new_x)):
      new_matrix[i][j]=new_x[j]-new_x[i]
  new_matrix=np.array(new_matrix)
  original_shape=new_matrix.shape[0]
  new_matrix=np.reshape(new_matrix,newshape=(-1,new_matrix.shape[-1]))
  
  new_matrix=(new_matrix-np.min(new_matrix,axis=0))/(np.max(new_matrix,axis=0)-np.min(new_matrix,axis=0))*255
  new_matrix=np.reshape(new_matrix,newshape=(original_shape,original_shape,-1))
  return new_matrix

def runRelativePositionMatrix():
    feas=np.load('/home/m21_huangzijun/pythonprojs/sichuan/Data2/pictures/feas.npy')
    new_feas=list()
    for sample in feas:
        new_sample= relativePositionMatrix(sample,k=2)
        new_feas.append(new_sample)
    new_feas=np.array(new_feas)
    np.save('/home/m21_huangzijun/pythonprojs/sichuan/Data2/pictures/pictures.npy',new_feas,allow_pickle=True)
    pass



if __name__ == "__main__":
    test5()
