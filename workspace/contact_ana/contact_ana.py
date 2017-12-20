import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('d:/resources/contact_ana/cmu_name_1219.csv', index_col=0)
bad = pd.read_csv('d:/resources/contact_ana/bad.csv')


def trans_list(df, col_name):
    df_name_list = df[col_name]
    return df_name_list.replace('\'', '').replace(' ', '')[1:-1].split(',')


df['name_list'] = df.apply(trans_list, axis=1, col_name='name_list')
df['clean_name_list'] = df.apply(trans_list, axis=1, col_name='clean_name_list')

df_merge = pd.merge(df, bad, on='mem_id', how='left')

df_merge.loc[df_merge.bad > 0, 'bad'] = 1


def max_name_radio(name_list):
    max_key = None
    max_count = 0
    count = {}
    for name in name_list:
        if len(name) == 0:
            continue
        key = name[0]
        if key == '小':
            if len(name) == 1:
                continue
            else:
                key = name[1]
        if key in count.keys():
            count[key] += 1
        else:
            count[key] = 1
        if count[key] > max_count:
            max_key = key
            max_count = count[key]
    return [max_key, max_count / len(name_list)]


key_ratio = np.array(list(df_merge['clean_name_list'].apply(max_name_radio).values))

df_merge['max_last_name_ratio'] = key_ratio[:, 1]
df_merge['max_last_name_key'] = key_ratio[:, 0]

# df_merge['clean_name_len'] = df_merge['clean_name'].apply(lambda x: len(x))
df_merge['valid_ratio'] = df_merge['clean_len'] / df_merge['list_len']

df_merge_to_save = df_merge[
    ['mem_id', 'list_len', 'clean_len', 'valid_ratio', 'max_last_name_key',
     'max_last_name_ratio',
     'bad']]

df_merge_to_save.to_csv('d:/resources/contact_ana/after_clean_1219.csv', encoding='utf8')

names = df_merge.groupby('max_last_name_key').agg(lambda x: x.count()).copy()
names = names[['mem_id']]
names.columns = ['freq']
names = names.sort_values(by='freq', ascending=False)

df_bad = df_merge[df_merge.bad != 0]
df_good = df_merge[df_merge.bad == 0]

g = sns.FacetGrid(df_merge, hue='bad')
g.map(sns.kdeplot, 'name_len').add_legend()
