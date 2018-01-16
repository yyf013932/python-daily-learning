# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 15:20:47 2017

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

stop_words = pd.read_csv('d:/tools/nlp/stopwords.csv', encoding='gb2312', header=None)
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

"""
"""

df_use['get_send'] = df_use['get_send'].astype(np.int32)
df_use['good_bad'] = df_use['good_bad'].astype(np.int32)
df_use_group = df_use[df_use['get_send'] < 3]
count_df = pd.DataFrame(
    df_use_group['text'].groupby([df_use_group['good_bad'], df_use_group['get_send']]).count())
count_df.reset_index(inplace=True)
unique_mem_df = pd.DataFrame(
    df_use_group['mem_id'].groupby([df_use['good_bad'], df_use['get_send']]).unique()).reset_index()
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
key_word_df = pd.read_csv('d:/int2word.csv', header=None, encoding='gbk')
key_word_df.columns = ['num', 'word']
key_word_set = set(key_word_df['word'])

count_matrix = np.zeros(shape=[word_fre_df.shape[0], len(key_word_set)])

bianhao = []
keyword_list = []
for i, j in enumerate(key_word_set):
    bianhao.append(i)
    keyword_list.append(j)

key_word_df_new = pd.DataFrame({'bianhao': bianhao, 'word': keyword_list})

word2int = {}
for i, j in key_word_df_new.values:
    word2int[j] = i

sms_word_count_narray = word_fre_df['out_stop_sms_all_list'].values
for j in range(len(sms_word_count_narray)):
    for i in word2int.keys():
        if i in sms_word_count_narray[j]:
            count_matrix[j, word2int[i]] = 1

count_matrix_df = pd.DataFrame(count_matrix)

count_matrix_df.columns = key_word_df_new['word'].values

statistic_word_df = pd.concat([word_fre_df, count_matrix_df], axis=1)

# statistic_word_df.to_csv('d:/s_word_df.csv',header=True,encoding='gb18030')
# statistic_word_df=pd.read_csv('d:/s_word_df.csv',header=0,encoding='gb18030',lineterminator='\n',index_col=0)

statistic_word_df.rename(columns={'没事\r': '没事'}, inplace=True)

# 一个一个词看
# tem_df1=pd.DataFrame(statistic_word_df['吃饭'].groupby([statistic_word_df['mem_id'],statistic_word_df['good_bad'],\
# statistic_word_df['get_send']]).mean()).reset_index()

# tem_df1[(tem_df1['good_bad']==1)&(tem_df1['get_send']==1)&tem_df1['吃饭']>0].shape

"""
good_all:5714 收3545 发2194  bad_all:1064   收664  发400
"""

pro_dict = {'gg': 3545, 'gs': 2194, 'bg': 664, 'bs': 400}  # 第一位好人坏人，第二位收发
pro_list = np.array([3545, 2194, 664, 400])


def get_propo1(word):
    """
    输入词
    输出有各类短信中 含有这个词 的短信占比
    """
    tem_df = pd.DataFrame(
        statistic_word_df[word].groupby([statistic_word_df['mem_id'], statistic_word_df['good_bad'], \
                                         statistic_word_df['get_send']]).mean()).reset_index()
    gg = tem_df[(tem_df['good_bad'] == 0) & (tem_df['get_send'] == 1) & (tem_df[word] > 0)].shape[0]
    gs = tem_df[(tem_df['good_bad'] == 0) & (tem_df['get_send'] == 2) & (tem_df[word] > 0)].shape[0]
    bg = tem_df[(tem_df['good_bad'] == 1) & (tem_df['get_send'] == 1) & (tem_df[word] > 0)].shape[0]
    bs = tem_df[(tem_df['good_bad'] == 1) & (tem_df['get_send'] == 2) & (tem_df[word] > 0)].shape[0]
    g_all = tem_df[(tem_df['good_bad'] == 0) & (tem_df[word] > 0)].shape[0]
    b_all = tem_df[(tem_df['good_bad'] == 1) & (tem_df[word] > 0)].shape[0]
    global pro_list
    return np.array([gg, gs, bg, bs, g_all, b_all]) / np.array([3545, 2194, 664, 400, 5739, 1064])


word_column = []
pro_column = []
for i in word2int.keys():
    word_column.append(i)
    pro_column.append(get_propo1(i))

pro_df1 = pd.concat([pd.DataFrame(word_column), pd.DataFrame(pro_column)], axis=1)
pro_df1.columns = ['word', 'gg', 'gs', 'bg', 'bs', 'g_all', 'b_all']
pro_df1['gg-bg'] = np.absolute(pro_df1['gg'] - pro_df1['bg'])
pro_df1['gs-bs'] = np.absolute(pro_df1['gs'] - pro_df1['bs'])
pro_df1['gall-ball'] = np.absolute(pro_df1['g_all'] - pro_df1['b_all'])


def get_propo2(word):
    """
    输入词
    输出 按member_id  get_send  good_bad group by 含有这个词 的短信占比的平均值
    """
    tem_df = pd.DataFrame(
        statistic_word_df[word].groupby([statistic_word_df['mem_id'], statistic_word_df['good_bad'], \
                                         statistic_word_df['get_send']]).mean()).reset_index()
    gg = tem_df[(tem_df['good_bad'] == 0) & (tem_df['get_send'] == 1)][word].mean()
    gs = tem_df[(tem_df['good_bad'] == 0) & (tem_df['get_send'] == 2)][word].mean()
    bg = tem_df[(tem_df['good_bad'] == 1) & (tem_df['get_send'] == 1)][word].mean()
    bs = tem_df[(tem_df['good_bad'] == 1) & (tem_df['get_send'] == 2)][word].mean()
    g_all = tem_df[tem_df['good_bad'] == 0][word].mean()
    b_all = tem_df[tem_df['good_bad'] == 1][word].mean()
    return np.array([gg, gs, bg, bs, g_all, b_all])


word_column_m = []
pro_column_m = []
for i in word2int.keys():
    word_column_m.append(i)
    pro_column_m.append(get_propo2(i))

pro_df2 = pd.concat([pd.DataFrame(word_column_m), pd.DataFrame(pro_column_m)], axis=1)
pro_df2.columns = ['word', 'gg', 'gs', 'bg', 'bs', 'g_all', 'b_all']
pro_df2['gg-bg'] = np.absolute(pro_df2['gg'] - pro_df2['bg'])
pro_df2['gs-bs'] = np.absolute(pro_df2['gs'] - pro_df2['bs'])
pro_df2['g_all-b_all'] = np.absolute(pro_df2['g_all'] - pro_df2['b_all'])
pro_df2[pro_df2.columns[1:]] = pro_df2[pro_df2.columns[1:]].apply(lambda x: x * 100)

sw1 = pro_df1[pro_df1['gall-ball'] > 0.025]['word']
sw2 = pro_df1[pro_df1['gg-bg'] > 0.04]['word']
sw3 = pro_df1[pro_df1['gs-bs'] > 0.025]['word']
sw4 = pro_df2[pro_df2['gs-bs'] > 0.4]['word']
sw5 = pro_df2[pro_df2['gg-bg'] > 1.3]['word']
sw6 = pro_df2[pro_df2['g_all-b_all'] > 0.75]['word']

select_word1 = set(list(sw1.values))
select_word2 = set(list(sw2.values))
select_word3 = set(list(sw3.values))
select_word4 = set(list(sw4.values))
select_word5 = set(list(sw5.values))
select_word6 = set(list(sw6.values))

select_word_df = pd.DataFrame(list(select_word1 | select_word2 | select_word5 | select_word6))
select_word_df.to_csv('d:/select_word_raw.csv')

select_word_df_send = pd.DataFrame(list(select_word3 | select_word4))
select_word_df_send.to_csv('d:/select_word_send_raw.csv')
