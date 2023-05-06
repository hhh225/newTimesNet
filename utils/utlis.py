# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 19:49:06 2022

@author: Administrator
"""
from contextlib import contextmanager
from time import time,sleep

@contextmanager
def timespend(str1):
    t1=time()
    yield
    t2=time()
    print('{} spend time:{}'.format(str1,t2-t1))

if __name__=="__main__":
    with timespend('waiting') as t:
        sleep(15)
        