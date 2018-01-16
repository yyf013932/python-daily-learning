import numpy as np
import pandas as pd

df = pd.read_csv('d:/resources/sms_analysis/yky_sms.csv', header=None).dropna()

df.columns = ['mem_id', 'bad', 'get_receive', 'mess']

df_notna = df[df.mess != '\\N']

key_words = ['借钱', '贷款', '金额', '人民币', '余额']


def filter_mess(df, keywords):
    content = df['mess']
    for k in keywords:
        if content.__contains__(k):
            return content
    return None


df_filter = df.apply(filter_mess, axis=1, keywords=['建设银行']).dropna()
df_valid = df.loc[df_filter.index]
