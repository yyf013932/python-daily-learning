# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 18:10:30 2017

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

stop_words = pd.read_csv('d:/resources/sms_analysis/stopwords.csv', encoding='gb2312', header=None)
stop_words_list = list(np.squeeze(stop_words.values, axis=1))
stop_words_set = set(stop_words_list)
stop_words_set.add('\\')
stop_words_set.add('\r')
df = pd.read_csv('d:/yky_sms.csv', header=None)
df.columns = ['mem_id', 'good_bad', 'get_send', 'text']
df_use = df[df['get_send'] != '\\N']
df_use.reset_index(inplace=True)


def clean_txt(x):
    m = x.strip()
    m = re.sub(
        '[0-9a-zA-Z\n/！，!。‘’“”;？,「」①●ω《~》∩"；：、〈〉〖〗_\?#&=…<>\-\+\*%％￥|\$@『』～·:\\（）→\(\)—【】……{}\.]',
        '', m)
    m = m.replace('[', '')
    m = m.replace(']', '')
    m = re.sub('[ ]+', ' ', m)
    return m.strip()


df_use['text_type'] = df_use['text'].apply(lambda x: str(type(x)))
df_used = df_use[df_use['text_type'] != "<class 'float'>"]
df_used['clean_text'] = df_used['text'].apply(
    lambda x: clean_txt(' '.join(jieba.cut(x, cut_all=False))))
df_used['is_num'] = df_used['text'].apply(lambda x: x.isdigit())
df_used2 = df_used[df_used['is_num'] == False]
train = df_used2[['mem_id', 'good_bad', 'get_send', 'clean_text']]
train2 = train[train['clean_text'] != '']
train2.reset_index(inplace=True)
del train2['index']
train2['sms_len'] = train2['clean_text'].apply(lambda x: len(x))
train_ = train2[train2['sms_len'] >= 7]
train_.to_csv('d:/sms_clean/clean_txt.csv')
sms_value = train_['clean_text'].values

out_stopwords_sms = []
for i in range(len(sms_value)):
    temp_array = []
    for j in sms_value[i].split(' '):
        if j not in stop_words_set:
            temp_array.append(j)
    out_stopwords_sms.append(temp_array)

train_['out_stop_sms'] = out_stopwords_sms
train_['out_stop_sms_all_list'] = train_['out_stop_sms'].apply(lambda x: ' '.join(x))

grouped = train_.groupby([train_['mem_id'], train_['get_send'], train_['good_bad']])
name_list = []

for name, group in grouped:
    name_list.append(name)
name_list_df = pd.DataFrame(name_list, columns=['mem_id', 'get_send', 'good_bad'])

all_sms_flat = []
group_length = len(list(grouped))
list_grouped = list(grouped)
for i in range(group_length):
    all_sms_flat.append(
        ' '.join(list(np.array(list_grouped[i][1]['out_stop_sms_all_list']).flatten())))

name_list_df['flat_sms'] = all_sms_flat

""" 
保持/读取 整理好的按memeber_id,get_send,good_bad groupby的文本

name_list_df.to_csv('D:/t11.csv',index=False,header=True,encoding='gb18030',lineterminator='\n')
name_list_df=pd.read_csv('D:/t11.csv',index_col=None,header=0,encoding='gb18030',lineterminator='\n') 
name_list_df.rename(columns={'flat_sms\r' : 'flat_sms'},inplace=True)

"""

name_list_df['get_send'] = name_list_df['get_send'].astype(np.int32)
name_list_df['good_bad'] = name_list_df['good_bad'].astype(np.int32)
key_words = []
sms_narray = name_list_df['flat_sms'].values
topK = 30
for i in range(len(sms_narray)):
    key_words.append(jieba.analyse.extract_tags(sms_narray[i], topK=topK))

key_words_all = []
for i in range(len(key_words)):
    for j in range(len(key_words[i])):
        key_words_all.append(key_words[i][j])

key_words_all_df = pd.DataFrame(collections.Counter(key_words_all).most_common(),
                                columns=['word', 'times'])

"""
"""
key_words11 = []
sms11_narray = name_list_df[(name_list_df['get_send'] == 1) & (name_list_df['good_bad'] == 1)][
    'flat_sms'].values
for i in range(len(sms11_narray)):
    key_words11.append(jieba.analyse.extract_tags(sms11_narray[i], topK=topK))
key_words_all11 = []
for i in range(len(key_words11)):
    for j in range(len(key_words11[i])):
        key_words_all11.append(key_words11[i][j])

key_words_all_df11 = pd.DataFrame(collections.Counter(key_words_all11).most_common(),
                                  columns=['word11', 'times'])

"""
"""
key_words21 = []
sms21_narray = name_list_df[(name_list_df['get_send'] == 2) & (name_list_df['good_bad'] == 1)][
    'flat_sms'].values
for i in range(len(sms21_narray)):
    key_words21.append(jieba.analyse.extract_tags(sms21_narray[i], topK=topK))
key_words_all21 = []
for i in range(len(key_words21)):
    for j in range(len(key_words21[i])):
        key_words_all21.append(key_words21[i][j])

key_words_all_df21 = pd.DataFrame(collections.Counter(key_words_all21).most_common(),
                                  columns=['word21', 'times'])

"""
"""
key_words10 = []
sms10_narray = name_list_df[(name_list_df['get_send'] == 1) & (name_list_df['good_bad'] == 0)][
    'flat_sms'].values
for i in range(len(sms10_narray)):
    key_words10.append(jieba.analyse.extract_tags(sms10_narray[i], topK=topK))
key_words_all10 = []
for i in range(len(key_words10)):
    for j in range(len(key_words10[i])):
        key_words_all10.append(key_words10[i][j])

key_words_all_df10 = pd.DataFrame(collections.Counter(key_words_all10).most_common(),
                                  columns=['word10', 'times'])

"""
"""
key_words20 = []
sms20_narray = name_list_df[(name_list_df['get_send'] == 2) & (name_list_df['good_bad'] == 0)][
    'flat_sms'].values
for i in range(len(sms20_narray)):
    key_words20.append(jieba.analyse.extract_tags(sms20_narray[i], topK=topK))
key_words_all20 = []
for i in range(len(key_words20)):
    for j in range(len(key_words20[i])):
        key_words_all20.append(key_words20[i][j])

key_words_all_df20 = pd.DataFrame(collections.Counter(key_words_all20).most_common(),
                                  columns=['word20', 'times'])

n_word = 50
key_words_analysis = pd.concat([key_words_all_df.head(n_word), key_words_all_df11.head(n_word),
                                key_words_all_df21.head(n_word),
                                key_words_all_df10.head(n_word), key_words_all_df20.head(n_word)],
                               axis=1, ignore_index=True)

key_words_analysis.columns = ['get_good_all', 'times', 'get_good_11', \
                              'times', 'get_good_21', 'times', 'get_good_10', 'times',
                              'get_good_20', 'times']

if not os.path.exists('d:/word_important.csv'):
    key_words_analysis.to_csv('d:/word_important.csv')
else:
    key_words_analysis.to_csv(
        'd:/' + time.strftime("%Y-%m-%d", time.localtime()) + '_word_important.csv')

"""
"""

df_use['get_send'] = df_use['get_send'].astype(np.int32)
df_use['good_bad'] = df_use['good_bad'].astype(np.int32)
df_use_group = df_use[df_use['get_send'] < 3]
count_df = pd.DataFrame(
    df_use_group['text'].groupby([df_use_group['good_bad'], df_use_group['get_send']]).count())
count_df.reset_index(inplace=True)
unique_mem_df = pd.DataFrame(df_use_group['mem_id'].groupby(
    [df_use_group['good_bad'], df_use_group['get_send']]).unique()).reset_index()
unique_mem_df['mem_num'] = unique_mem_df['mem_id'].apply(lambda x: len(x))
count_df['mean_num'] = count_df['text'] / unique_mem_df['mem_num']
"""
好人收平均收150条，坏人收120条，发差异不大。
"""
train_['get_send'] = train_['get_send'].astype(np.int32)
train_['good_bad'] = train_['good_bad'].astype(np.int32)
train_count = train_[train_['get_send'] < 3]
count_df_mem = pd.DataFrame(df_use_group['text'].groupby(
    [df_use_group['mem_id'], df_use_group['get_send'],
     df_use_group['good_bad']]).count()).reset_index()

count_df_mem[(count_df_mem['get_send'] == 1) & (count_df_mem['good_bad'] == 1)]['text']

"""
好人、坏人短信平均长度差异不大。
"""
train2['sms_len'].groupby([train2['good_bad'], train2['get_send']]).mean()

train_.ix[0, 'out_stop_sms_all_list']

train_count_use = pd.DataFrame(train_count['out_stop_sms_all_list'].groupby([train_count['mem_id'], \
                                                                             train_count[
                                                                                 'good_bad'],
                                                                             train_count[
                                                                                 'get_send']]).count()).reset_index()

train_count_use.columns = ['mem_id', 'good_bad', 'get_send', 'sms_num']

word_fre_df = pd.merge(train_count, train_count_use, how='inner',
                       on=['mem_id', 'good_bad', 'get_send'])
'''
'out_stop_sms_all_list'
'''
