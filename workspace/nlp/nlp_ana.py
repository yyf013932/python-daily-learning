import pandas as pd
import numpy as np
import jieba

df = pd.read_csv('d:/resources/NLP/training.csv', header=None)
f = open('d:/resources/NLP/stopwords.csv', encoding='utf8')
stopwords = [w.strip() for w in f.readlines()]
f.close()


def cut_word(sentence):
    return [w for w in jieba.cut(sentence) if w not in stopwords]


df.columns = ['type', 'content']

df['cut_content'] = df['content'].apply(cut_word)

word_dic = []


def append_if_not_exist(to_append, data):
    if type(data) == list:
        [append_if_not_exist(to_append, w) for w in data]
    else:
        if data not in to_append:
            to_append.append(data)


df['cut_content'].apply(append_if_not_exist)

document_list = [document for document in df.cut_content.values]

from collections import defaultdict
import math


class chi_feature_select:
    """
    卡方统计量进行特征选取
    """

    def __init__(self, documents, labels):
        pass


class TF_IDF:
    """
    用于计算TF_IDF值
    """

    def __init__(self, documents):
        self.documents = documents
        self.document_len = len(documents)
        self.word_len = 0
        self.TF = []
        self.IDF = defaultdict(int)
        self.TF_IDF = []
        for document in documents:
            self.word_len += len(document)
            tem_set = set()
            tem_dic = defaultdict(int)
            for w in document:
                tem_dic[w] += 1
                tem_set.add(w)
            self.TF.append(tem_dic)
            for w in tem_set:
                self.IDF[w] += 1
        for k in self.IDF:
            self.IDF[k] = math.log2(self.document_len / self.IDF[k])
        for dic in self.TF:
            tem_dic = defaultdict(int)
            for k in dic.keys():
                tem_dic[k] = dic[k] * self.IDF[k]
            self.TF_IDF.append(tem_dic)


tf_idf = TF_IDF(document_list)


