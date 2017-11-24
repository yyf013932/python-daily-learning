# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 08:26:46 2017

@author: chriszhang
"""

import pandas as pd
import numpy as np
import re
import jieba
import jieba.analyse
import csv
import collections
import os
import time
import matplotlib.pyplot as plt


  
model_df1=pd.read_csv('d:/s_word_df.csv',header=0,encoding='gb18030',lineterminator='\n',index_col=0)    

model_df1.rename(columns={'没事\r' : '没事'},inplace=True)


model_df2=pd.read_csv('d:/10m/s_word_df.csv',header=0,encoding='gb18030',lineterminator='\n',index_col=0)    

model_df2.rename(columns={'没事\r' : '没事'},inplace=True)

get_key_words=['手机','号码','如非','验证码','查询','人民币','账户','尾号','微信','储蓄卡','还款','搜索','现金',\
'中国农业银行','中国移动','操作','客户','短信','消费','支出','余额','信用']


send_key_words=['手机','收到','你好','打电话','密码','回来','地址','短信','快递','信息',\
'马上','微信','您好','有事','谢谢','下午','麻烦']


co_occu_word2=['现金&还款','微信&还款','操作&还款','马上&微信','查询&验证码','消费&验证码','微信&马上',\
'微信&搜索','微信&马上','操作&账户','还款&账户','消费&信用','中国农业银行&还款']



co_occu_word3=['还款&微信&客户','操作&还款&短信','消费&信用&查询','客户&人民币&还款','消费&尾号&信用',\
'账户&如非&还款','还款&手机&信用','验证码&如非&操作','微信&还款&余额']

"""



def pad2_matrix(df):
    """
    输入df
    输出2个词的词共现矩阵
    """
    pad2_mat=[]
    for index in range(df.shape[0]):
        c2_array=np.zeros(shape=[len(co_occu_word2)])
        for i in range(len(co_occu_word2)):
            if df.ix[index , co_occu_word2[i].split('&')[0]]==1. \
            and   df.ix[index , co_occu_word2[i].split('&')[1]]==1.:
                c2_array[i]=1.
        pad2_mat.append(c2_array)
    return np.array(pad2_mat)





def pad3_matrix(df):
    """
    输入df
    输出3个词的词共现矩阵
    """
    pad3_mat=[]
    for index in range(df.shape[0]):
        c3_array=np.zeros(shape=[len(co_occu_word3)])
        for i in range(len(co_occu_word3)):
            if df.ix[index , co_occu_word3[i].split('&')[0]]==1. \
            and   df.ix[index , co_occu_word3[i].split('&')[1]]==1.\
            and   df.ix[index , co_occu_word3[i].split('&')[2]]==1.:
                c3_array[i]=1.
        pad3_mat.append(c3_array)
    return np.array(pad3_mat)



m1_co2=pad2_matrix(model_df1)
m1_co3=pad3_matrix(model_df1)
m2_co2=pad2_matrix(model_df2)
m2_co3=pad3_matrix(model_df2)


m1_co2_df=pd.DataFrame(m1_co2)
m1_co3_df=pd.DataFrame(m1_co3)
m2_co2_df=pd.DataFrame(m2_co2)
m2_co3_df=pd.DataFrame(m2_co3)

m1_co2_df.columns=co_occu_word2
m1_co3_df.columns=co_occu_word3
m2_co2_df.columns=co_occu_word2
m2_co3_df.columns=co_occu_word3




model_df1=pd.concat([model_df1,m1_co2_df,m1_co3_df],axis=1)
model_df2=pd.concat([model_df2,m2_co2_df,m2_co3_df],axis=1)
































"""

m1_get=model_df1[get_key_words][model_df1['get_send']==1].groupby([model_df1['mem_id'],\
model_df1['good_bad']]).mean().reset_index()

m1_send=model_df1[send_key_words][model_df1['get_send']==2].groupby([model_df1['mem_id'],\
model_df1['good_bad']]).mean().reset_index()

m1_co2p=model_df1[co_occu_word2].groupby([model_df1['mem_id'],\
model_df1['good_bad']]).mean().reset_index()

m1_co3p=model_df1[co_occu_word3].groupby([model_df1['mem_id'],\
model_df1['good_bad']]).mean().reset_index()


m2_get=model_df2[get_key_words][model_df2['get_send']==1].groupby([model_df2['mem_id'],\
model_df2['good_bad']]).mean().reset_index()

m2_send=model_df2[send_key_words][model_df2['get_send']==2].groupby([model_df2['mem_id'],\
model_df2['good_bad']]).mean().reset_index()

m2_co2p=model_df2[co_occu_word2].groupby([model_df1['mem_id'],\
model_df1['good_bad']]).mean().reset_index()

m2_co3p=model_df2[co_occu_word3].groupby([model_df1['mem_id'],\
model_df1['good_bad']]).mean().reset_index()









m1_get_sms_num=model_df1['out_stop_sms'][model_df1['get_send']==1].groupby([model_df1['mem_id'],\
model_df1['good_bad']]).count().reset_index()
m1_get_sms_num.columns=['mem_id', 'good_bad', 'get_num']

m1_send_sms_num=model_df1['out_stop_sms'][model_df1['get_send']==2].groupby([model_df1['mem_id'],\
model_df1['good_bad']]).count().reset_index()
m1_send_sms_num.columns=['mem_id', 'good_bad', 'send_num']

m2_get_sms_num=model_df2['out_stop_sms'][model_df2['get_send']==1].groupby([model_df2['mem_id'],\
model_df2['good_bad']]).count().reset_index()
m2_get_sms_num.columns=['mem_id', 'good_bad', 'get_num']

m2_send_sms_num=model_df2['out_stop_sms'][model_df2['get_send']==2].groupby([model_df2['mem_id'],\
model_df2['good_bad']]).count().reset_index()
m2_send_sms_num.columns=['mem_id', 'good_bad', 'send_num']


m1_sms2num=pd.merge(m1_get_sms_num,m1_send_sms_num,how='inner',on=['mem_id','good_bad'])
m2_sms2num=pd.merge(m2_get_sms_num,m2_send_sms_num,how='inner',on=['mem_id','good_bad'])



m1_train=pd.merge(m1_sms2num,m1_get,how='inner',on=['mem_id','good_bad'])
m2_train=pd.merge(m2_sms2num,m2_get,how='inner',on=['mem_id','good_bad'])

m1_train2=pd.merge(m1_train,m1_send,how='inner',on=['mem_id','good_bad'])
m2_train2=pd.merge(m2_train,m2_send,how='inner',on=['mem_id','good_bad'])

"""
词有_x的是收短信词
词有_y的是发短信词
"""


all_train=pd.concat([m2_train2.iloc[:,1:],m1_train2.iloc[:,1:]],ignore_index=True)

#all_train=pd.concat([m2_train2,m1_train2],ignore_index=True)

"""
如下是仅考虑收短信
"""

m1_train_get=pd.merge(m1_get_sms_num,m1_get,how='inner',on=['mem_id','good_bad'])
m2_train_get=pd.merge(m2_get_sms_num,m2_get,how='inner',on=['mem_id','good_bad'])

all_train_get=pd.concat([m2_train_get.iloc[:,1:],m1_train_get.iloc[:,1:]],ignore_index=True)

#all_train_get=pd.concat([m2_train_get,m1_train_get],ignore_index=True)

"""
保存/读取

all_train.to_csv('d:/sms_model/all_train.csv')
all_train=pd.read_csv('d:/sms_model/all_train.csv',header=0,encoding='gb18030',index_col=0)


all_train_get.to_csv('d:/sms_model/all_train_get.csv')
all_train_get=pd.read_csv('d:/sms_model/all_train_get.csv',header=0,encoding='gb18030',index_col=0)


"""
all_train['get/send']=all_train['get_num']/all_train['send_num']
all_train['log_get/send']=np.log(all_train['get/send'])

y_all=all_train.ix[:,0]  #考虑收发
x_all=all_train.ix[:,1:]
n_x_all=(x_all-x_all.mean())/x_all.std()


y_all_get=all_train_get.ix[:,0]   #只考虑收
x_all_get=all_train_get.ix[:,1:]
n_x_all_get=(x_all_get-x_all_get.mean())/x_all_get.std()



del_feat=['人民币','尾号','操作','支出']

x_all_del_feat=x_all[list(set(x_all.columns)-set(del_feat))]    #考虑收发 去除相关性词
n_x_all_del_feat=(x_all_del_feat-x_all_del_feat.mean())/x_all_del_feat.std()


x_all_get_del_feat=x_all_get[list(set(x_all_get.columns)-set(del_feat))]  #  考虑收 去除相关性词
n_x_all_get_del_feat=(x_all_get_del_feat-x_all_get_del_feat.mean())/x_all_get_del_feat.std()

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
from sklearn.model_selection import GridSearchCV





def get_best_para( x , y , c_max = 1.4 , cv =4 ,metric_n='roc_auc'):
    estimator=LogisticRegression(class_weight='balanced',penalty='l2')
    param_grid = {'solver': ['newton-cg', 'lbfgs', 'liblinear'],'C': np.arange(0.1,c_max,0.05),\
    'fit_intercept':[True,False],'intercept_scaling':[0.5,1.,1.2,1.5,2]}
    clf = GridSearchCV(estimator= estimator, scoring=metric_n, param_grid=param_grid,n_jobs=-1 , cv=cv ) 
    clf.fit( x , y )
    return clf.best_estimator_ , clf.best_score_
    #return clf.grid_scores_, clf.best_params_, clf.best_score_






col=[]
zero_mean=[]
for i in all_train.columns:
    col.append(i)
    zero_mean.append((all_train[i]>0).mean())

zero_df=pd.DataFrame({'col':col,'zero_mean':zero_mean})    
zero_df[zero_df['zero_mean']>0.5].shape
zero_df[zero_df['zero_mean']>0.5]['col'].values

list_col=list(zero_df[zero_df['zero_mean']>0.5]['col'].values)
list_col_get=['get_num', '手机', '如非', '验证码', '查询', '账户', '尾号', '微信', '还款', '搜索', '现金',\
'中国移动', '操作', '客户', '短信', '消费', '余额', '信用']



word_list=['手机_x', '号码', '如非', '验证码', '查询', '人民币', '账户',
       '尾号', '微信_x', '储蓄卡', '还款', '搜索', '现金', '中国农业银行', '中国移动', '操作', '客户',
       '短信_x', '消费', '支出', '余额', '信用', '手机_y', '收到', '你好', '打电话', '密码', '回来',
       '地址', '短信_y', '快递', '信息', '马上', '微信_y', '您好', '有事', '谢谢', '下午', '麻烦']


word_list_get=['手机', '号码', '如非', '验证码', '查询', '人民币', '账户',
       '尾号', '微信', '储蓄卡', '还款', '搜索', '现金', '中国农业银行', '中国移动', '操作', '客户',
       '短信', '消费', '支出', '余额', '信用']


best_coef=np.array([-0.15852199,  0.03869031, -0.06112389,  0.1127114 ,  0.05345079,
         0.03198817,  0.07682791, -0.02635987,  0.03916395, -0.05481987,
        -0.02376087, -0.04765489,  0.05496105,  0.01168536, -0.00784705,
        -0.03425504,  0.12143126,  0.07842589, -0.03852315, -0.02499455,
        -0.10889325, -0.05777167,  0.02274014,  0.1081524 ,  0.08047573,
        -0.08255566, -0.08700037, -0.00804214, -0.00394867, -0.24907426,
        -0.21026199,  0.09704844, -0.1334906 ,  0.10607999,  0.02141754,
         0.03852485, -0.06260437,  0.00593282, -0.02285462, -0.07835697,
        -0.09578323, -0.02866589, -0.01698912])




extra_x_all=x_all
extra_x_all['44']=extra_x_all['现金']*extra_x_all['还款']
extra_x_all['45']=extra_x_all['微信_x']*extra_x_all['还款']
extra_x_all['46']=extra_x_all['操作']*extra_x_all['还款']
extra_x_all['47']=extra_x_all['马上']*extra_x_all['微信_y']
extra_x_all['48']=extra_x_all['查询']*extra_x_all['验证码']
extra_x_all['49']=extra_x_all['消费']*extra_x_all['验证码']
extra_x_all['50']=extra_x_all['微信_y']*extra_x_all['马上']
extra_x_all['51']=extra_x_all['微信_x']*extra_x_all['搜索']
extra_x_all['52']=extra_x_all['微信_x']*extra_x_all['马上']
extra_x_all['52']=extra_x_all['操作']*extra_x_all['账户']
extra_x_all['53']=extra_x_all['还款']*extra_x_all['账户']
extra_x_all['54']=extra_x_all['消费']*extra_x_all['信用']

"""
pd.concat([y_all,x_all],axis=1)
train_data=pd.read_csv('d:/sms_model/train11-15.csv',header=0,index_col=0,encoding='gb18030')
"""



extra_best_coef=np.array([ -9.40472736e-04,   1.56589486e-03,  -5.43006094e-01,
          8.07983187e-01,   5.37393321e-01,   6.27831087e-01,
          5.15361134e-01,  -1.45391884e-02,   6.44487663e-02,
         -2.01627912e-01,   4.14926753e-02,  -6.37178967e-01,
          7.18402189e-01,   2.82545468e-01,   1.06199605e-01,
          5.45073367e-02,   7.30878363e-01,   6.98960851e-01,
         -6.96464773e-02,  -1.75093108e-01,  -6.47645434e-01,
         -7.35952576e-01,  -7.51496277e-02,   8.94044963e-01,
          7.32061742e-01,  -5.23001128e-01,  -9.72130639e-01,
         -1.25606636e-01,   1.56132021e-02,  -1.72938024e+00,
         -1.17273513e+00,   8.08300461e-01,  -9.21448244e-01,
          1.02281608e+00,   2.52555859e-01,   3.03274748e-01,
         -7.16124892e-01,   9.03170786e-02,  -7.49181902e-01,
         -6.44832635e-01,  -7.34737234e-01,  -3.51740769e-04,
         -4.31739843e-03,   3.14964421e-02,   8.04721375e-02,
          5.46168679e-02,  -8.76484735e-03,   1.14517211e-01,
          1.33989804e-01,  -8.76484735e-03,   2.60205339e-02,
          9.83138594e-02])
          
          
extra_best_coef_recall=np.array([ -9.81760004e-04,   8.35635667e-04,  -2.00310645e-01,
          4.64063279e-01,   3.05902709e-01,   3.62330297e-01,
          3.88232815e-01,  -8.50308603e-02,   1.28745829e-02,
         -2.13343890e-01,   8.76488218e-02,  -3.19582680e-01,
          2.77322371e-01,   2.02264975e-01,   3.22943610e-02,
         -9.60542905e-03,   5.46064751e-01,   3.92956550e-01,
          1.34530144e-01,  -5.33256006e-02,  -3.39611085e-01,
         -3.82352917e-01,  -1.17867398e-01,   4.08354074e-01,
          3.21584073e-01,  -2.56673461e-01,  -4.08864080e-01,
         -4.16156527e-02,   6.89093949e-03,  -8.00326857e-01,
         -5.70305427e-01,   3.81652555e-01,  -4.52904150e-01,
          4.72893374e-01,   1.29160318e-01,   1.49243233e-01,
         -3.11705552e-01,   4.22184282e-02,  -2.95030748e-01,
         -3.14594899e-01,  -3.52580844e-01,  -2.75047378e-04,
          1.52296258e-02,   6.92236015e-03,   3.29295334e-02,
          2.55614094e-02,  -5.77965696e-03,   6.84483880e-02,
          8.17517647e-02,  -5.77965696e-03,   4.34095901e-02,
          5.73699773e-02])         
          

























