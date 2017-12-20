import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sbn

login_df = pd.read_csv('d:/resources/jdjr/t_login.csv', dtype={'log_id': 'str'})
# trade_df = pd.read_csv('d:/resources/jdjr/t_trade.csv')
trade_df = pd.read_csv('d:/resources/jdjr/t_trade_withlog.csv', dtype={'log_id': 'str'})


def cal_day_time(day_time):
    l = day_time.split(':')
    return int(l[0]) * 60 + int(l[1])


# 转换日期格式
def trans_time(df, col_name, pre_fix):
    df[pre_fix + 'date'] = df[col_name].apply(lambda x: x.split(' ')[0])
    df[pre_fix + 'day_time'] = df[col_name].apply(lambda x: cal_day_time(x.split(' ')[1]))


trans_time(login_df, 'time', 'login_')
trans_time(trade_df, 'time', 'trade_')

'''
# 将交易和登录行为映射（取最近的一次登录信息）
i = 0
tl_id = []
for r_id, row in trade_df.iterrows():
    if i % 500 == 0:
        print(i)
    i += 1
    a_id = login_df[login_df.id == row.id]
    a_id = a_id[a_id.result == 1]
    a = a_id[a_id.time < row.time]
    a = a.sort_values(by='time', ascending=False)
    if a.size == 0:
        tl_id.append(None)
    else:
        tl_id.append(a.iloc[0].log_id)

trade_df['log_id'] = tl_id
'''

# 只使用有登录记录对应的风险行为信息
# 有风险的共计3643条记录，有对应登录记录的3572条记录
tl_merge = pd.merge(trade_df, login_df.drop(['id'], axis=1), on='log_id')

tl_merge_risk = tl_merge[tl_merge.is_risk == 1]

tl_merge_norisk = tl_merge[tl_merge.is_risk == 0]

# 训练数据
all_data = tl_merge


def boolean2int(v):
    if v:
        return 1
    return 0


from datetime import *
import time


def cal_interval(df):
    login_date = df['time_y']
    trade_date = df['time_x']
    login_datetime = datetime.strptime(login_date, "%Y-%m-%d %H:%M:%S")
    trade_datetime = datetime.strptime(trade_date, "%Y-%m-%d %H:%M:%S.0")
    interval = trade_datetime.timestamp() - login_datetime.timestamp()
    return interval / 60


all_data['is_scan'] = all_data['is_scan'].apply(boolean2int)
all_data['is_sec'] = all_data['is_sec'].apply(boolean2int)
# 计算交易和登录的时间差
all_data['trade_login_interval'] = [cal_interval(row) for r_id, row in all_data.iterrows()]

# 按时间排序
all_data = all_data.sort_values(by='time_x')

# 去除不必要的列
all_data.drop(['rowkey', 'id', 'time_x', 'time_y', 'timestamp'], inplace=True, axis=1)

train_data = all_data[all_data.trade_date < '2015-06-01']
test_data = all_data[all_data.trade_date >= '2015-06-01']

train_X = train_data.loc[:, ['trade_day_time', 'log_from', 'type', 'is_scan', 'is_sec',
                             'login_day_time', 'trade_login_interval']]
train_Y = train_data['is_risk']

test_X = test_data.loc[:, ['trade_day_time', 'log_from', 'type', 'is_scan', 'is_sec',
                           'login_day_time', 'trade_login_interval']]
test_Y = test_data['is_risk']

from sklearn.preprocessing import StandardScaler

st_scaler = StandardScaler().fit(train_X[['trade_login_interval']])
train_X['trade_login_interval'] = st_scaler.transform(train_X[['trade_login_interval']])
test_X['trade_login_interval'] = st_scaler.transform(test_X[['trade_login_interval']])

'''
查看数据分布
'''
# log_from 为16时风险较高
# type 为3 时风险高
# is_scan 为False时风险高
tl_merge.groupby(by=['log_from', 'is_risk']).apply(lambda x: x.size)

sbn.factorplot(x='type', y='is_risk', data=all_data)

# time_long 数据比较奇怪，有特别高的登录时间，准备删除此特征
# 凌晨的时候风险高
m = sbn.FacetGrid(data=all_data, hue='is_risk', aspect=3)
m.map(sbn.kdeplot, 'trade_login_interval').add_legend()
tl_merge.hist()
