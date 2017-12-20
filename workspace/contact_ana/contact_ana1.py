import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('d:/resources/contact_ana/after_clean_1219.csv', index_col=0)

df_bad = df[df.bad == 1]
df_good = df[df.bad == 0]

g = sns.FacetGrid(df, hue='bad')
g.map(sns.kdeplot, 'max_last_name_ratio').add_legend()


def print_ratio(df):
    good = len(df[df.bad == 0].mem_id.values)
    bad = len(df[df.bad == 1].mem_id.values)
    len_ = len(df.values)
    if len_ == 0:
        return 0, 0
    return good / len_, bad / len_


def get_bad_ratio(ratio):
    return print_ratio(df[df.max_last_name_ratio > ratio])[1]


x = np.linspace(0, 1, 100)
y = [get_bad_ratio(i) for i in x]
plt.plot(x[:-1], y[:-1])
