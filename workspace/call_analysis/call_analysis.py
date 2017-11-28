import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sbn

from sklearn.model_selection import cross_val_score, KFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn import linear_model
from sklearn import preprocessing
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier

data = pd.read_csv('d:/resources/call_mess/total_data.csv', encoding='utf-8')
data = data.drop('Unnamed: 0', axis=1)
'''
data = pd.read_csv('D:\\resources\\call_mess\\df_model.csv', encoding='utf8')
data = data.drop('Unnamed: 0', axis=1)

data.columns = ['mongth_len', 'pos_num_total', 'pos_call_total', 'nag_call_total', 'pos_time_total',
                'nag_time_total', 'mem_id', 'good_bad', 'set_len', 'pos_contact_time_total',
                'nag_contact_time_total', 'pos_contact_freq_total', 'nag_contact_freq_total',
                'pos_nag_inter_set', 'pos_nag_union_set', 'nag_num_total']

data = data.loc[:, ['mem_id', 'mongth_len', 'pos_num_total', 'pos_call_total', 'nag_call_total',
                    'pos_time_total', 'nag_time_total', 'set_len', 'pos_contact_time_total',
                    'nag_contact_time_total', 'pos_contact_freq_total', 'nag_contact_freq_total',
                    'pos_nag_inter_set', 'pos_nag_union_set', 'nag_num_total',
                    'nav_contact_num_total', 'pos_contact_num_total', 'good_bad']]
'''

'''
部分月份为0，使用其他特征进行拟合
估计的月份多数为5.5个月左右，方差较小
'''
data_month_zero = data[data.mongth_len == 0]
data_month_nozero = data[data.mongth_len != 0]
X = data_month_nozero.loc[:, 'pos_num_total':'nag_num_total']
y = data_month_nozero.loc[:, 'mongth_len'].values[:, np.newaxis]
pre_X = data_month_zero.loc[:, 'pos_num_total':'nag_num_total']
x_scaled = preprocessing.StandardScaler().fit(X)
X = x_scaled.transform(X)
pre_X = x_scaled.transform(pre_X)
reg = linear_model.Ridge(alpha=.5)
reg.fit(X, y)
pre_y = reg.predict(pre_X)
# 替换
data.loc[data['mongth_len'] == 0, 'mongth_len'] = pre_y

# 需要计算月均值的列
need_ave_cols = ['pos_call_total', 'nag_call_total', 'pos_time_total', 'nag_time_total',
                 'pos_contact_time_total', 'nag_contact_time_total', 'pos_contact_freq_total',
                 'nag_contact_freq_total']
for col in need_ave_cols:
    data[col + '_ave'] = data[col] / data['mongth_len']


# 将数据的高值截断，根据ratio
def drop_ex_value(data, cols, ratio):
    for col in cols:
        count = data[col].count()
        num_truncate = int(count * ratio)
        data_sort = data[col].sort_values()
        value_truncate = data_sort.iloc[-(num_truncate + 1)]
        data.loc[data[col] > value_truncate, col] = value_truncate
        # data[col].apply(lambda x: max(x, value_truncate))
        data.reindex()


# 截断高值，将0.1%上分位截断
drop_ex_value(data, ['pos_num_total', 'pos_call_total', 'nag_call_total',
                     'pos_time_total', 'nag_time_total', 'set_len', 'pos_contact_time_total',
                     'nag_contact_time_total', 'pos_contact_freq_total', 'nag_contact_freq_total',
                     'pos_nag_inter_set', 'pos_nag_union_set', 'nag_num_total'], 0.005)

# 可视化查看特征
g = sbn.FacetGrid(data, hue='good_bad')
# g.map(sbn.kdeplot, 'pos_call_total', shade=True)
g.map(sbn.kdeplot, 'pos_contact_time_total_ave', shade=True).add_legend()

train_X = data.loc[:, 'mongth_len':'nag_num_total']
train_Y = data.loc[:, 'good_bad']
tX_scaled = preprocessing.StandardScaler().fit(train_X)
train_X = tX_scaled.transform(train_X)


# 降维


def get_best_para_rf(x, y, c_max=2., cv=4, metric_n='roc_auc'):
    estimator = RandomForestClassifier(class_weight="balanced")
    param_grid = {'n_estimators': [50, 100, 200], 'max_features': ["sqrt", "log2", .8, 0.6, 0.4]}
    clf = GridSearchCV(estimator=estimator, scoring=metric_n, param_grid=param_grid, cv=cv)
    clf.fit(x, y)
    return clf.best_estimator_, clf.best_score_, clf.best_params_


be, bs, bp = get_best_para_rf(train_X, train_Y)
