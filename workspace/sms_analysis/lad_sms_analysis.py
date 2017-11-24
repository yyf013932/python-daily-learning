# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 09:49:18 2017

@author: chriszhang
"""
"""
中文unicode编码问题，暂时不能实现模型
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


stop_words=pd.read_csv('d:/tools/nlp/stopwords.csv',encoding='gb2312',header=None)
stop_words_list=list(np.squeeze(stop_words.values,axis=1))
stop_words_set=set(stop_words_list)
stop_words_set.add('\\')
stop_words_set.add('\r')
df=pd.read_csv('d:/yky_sms.csv',header=None)
df.columns=['mem_id','good_bad','get_send','text']
df_use=df[df['get_send']!='\\N']
df_use.reset_index(inplace=True)


def clean_txt(x):
    m=x.strip()
    m=re.sub('[0-9a-zA-Z\n/！，!。‘’“”;？,「」①●ω《~》∩"；：、〈〉〖〗_\?#&=…<>\-\+\*%％￥|\$@『』～·:\\（）→\(\)—【】……{}\.]','',m)
    m=m.replace('[','')
    m=m.replace(']','')
    m=re.sub('[ ]+',' ',m)  
    return m.strip()


df_use['text_type']=df_use['text'].apply(lambda x:str(type(x)))    
df_used=df_use[df_use['text_type']!="<class 'float'>"]   
df_used['clean_text']=df_used['text'].apply(lambda x:clean_txt(' '.join(jieba.cut(x,cut_all=False))))
df_used['is_num']=df_used['text'].apply(lambda x:x.isdigit())
df_used2=df_used[df_used['is_num']==False]
train=df_used2[['mem_id','good_bad','get_send','clean_text']]
train2=train[train['clean_text']!='']
train2.reset_index(inplace=True)
del train2['index']
train2['sms_len']=train2['clean_text'].apply(lambda x:len(x))
train_=train2[train2['sms_len']>=7]
train_.to_csv('d:/sms_clean/clean_txt.csv')
sms_value=train_['clean_text'].values

out_stopwords_sms=[]
for i in range(len(sms_value)):
    temp_array=[]
    for j in  sms_value[i].split(' '):
        if j not in stop_words_set:
            temp_array.append(j)
    out_stopwords_sms.append(temp_array)
    
train_['out_stop_sms']=out_stopwords_sms    
train_['out_stop_sms_all_list']=train_['out_stop_sms'].apply(lambda x:' '.join(x))


grouped=train_.groupby([train_['mem_id'],train_['get_send'],train_['good_bad']])    
name_list=[]   

for name, group in grouped:
    name_list.append(name)
name_list_df=pd.DataFrame(name_list,columns=['mem_id','get_send','good_bad'])

"""
all_sms_flat=[]
group_length=len(list(grouped))
list_grouped=list(grouped)
for i in range(group_length):
    all_sms_flat.append(' '.join(list(np.array(list_grouped[i][1]['out_stop_sms_all_list']).flatten())))
"""    
    
all_sms_word_array=[]
group_length=len(list(grouped))
list_grouped=list(grouped)  
for i in range(group_length):    
    all_sms_word_array.append(list(list_grouped[i][1]['out_stop_sms'].values))
    

group_length=len(list(grouped))
list_grouped=list(grouped)  
all_sms_word_array_lda=[]
for i in range(group_length):
    temp_array=list(list_grouped[i][1]['out_stop_sms_all_list'].values)
    temp_u=[]
    for j in range(len(temp_array)):
        temp_u.extend(temp_array[j].encode('unicode-escape'))
    all_sms_word_array_lda.append(temp_u)
    
    
    
    
       
name_list_df_lda=np.copy(name_list_df)
name_list_df_lda=pd.DataFrame(name_list_df_lda,columns=name_list_df.columns)   
name_list_df_lda['sms_word_array']=all_sms_word_array
name_list_df_lda['sms_word_array_lda']=all_sms_word_array_lda 
 

name_list_df_lda['get_send']=name_list_df['get_send'].astype(np.int32)
name_list_df_lda['good_bad']=name_list_df['good_bad'].astype(np.int32)


def get_lda_topic(x):
    """
    输入文本大列表，大列表的元素为小列表，小列表里每个元素为词，一个小列表为一段文本
    输出 lad前10个主题分布
    """
    word_count_dict = gensim.corpora.Dictionary(x)
    corpus = [ word_count_dict.doc2bow(x_i) for x_i in x ]
    lda = models.LdaModel(corpus=corpus, id2word=word_count_dict, num_topics=100)
    return lda.print_topics(10)
        

def get_ldawords(x):
    """
    输入：lda.print_topics(10)
    输出：关键词数量分布
    """
    topic_word_all=[]
    for i in range(len(x)):
        #every_topic_word=x[i][1].split(' + ')
        every_topic_word=re.findall(r'\d\.\d{3}\*"[a-z]+?"',x[i][1])   #中文  [\u4e00-\u9fa5]
        every_topic_word_list=[]
        for j in range(len(every_topic_word)):
            every_topic_word_list.append(every_topic_word[j][7:-1])
        topic_word_all.extend(every_topic_word_list)
        result=pd.DataFrame(collections.Counter(topic_word_all).most_common(),columns=['word','times'])
    return result.loc[:50]









