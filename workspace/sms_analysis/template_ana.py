import re
import pandas as pd
import numpy as np

money_re = '\d*,?\d{1,3}(\.\d+)?'

tempate_dict = {
    '【中国银行】': [
        [
            re.compile('^.*收入(\((.*)\))?(人民币|RMB)(' + money_re + ').*余额(' + money_re + ').*$'),
            [2, 4, 6]
        ],
        [
            re.compile(
                '^.*(支付|支取|消费)(\((.*)\))?(人民币|RMB)(' + money_re + ').*余额(' + money_re + ')?.*$'),
            [3, 5, 7]
        ]
    ],
    '【支付宝】': [
        [],
        [
            re.compile('^.*((在|通过)(.*))?((成功)?(付款|捐款|消费))(' + money_re + ').*('
                                                                         'you_can_not_match)?$'),
            [3, 7, 9]
        ],
    ],
    # TODO 信用卡未处理
    '【工商银行】': [
        [
            re.compile('^.*收入([(（](.*)[)）])?(' + money_re + ').*余额(' + money_re + ').*$'),
            [2, 3, 5]
        ],
        [
            re.compile(
                '^.*支出([(（](.*)[)）])?(' + money_re + ').*余额(' + money_re + ').*$'),
            [2, 3, 6]
        ]
    ],
    '【中国农业银行】': [
        [
            re.compile('^.*完成(.*)人民币(' + money_re + ').*余额(' + money_re + ').*$'),
            [1, 2, 4]
        ],
        [
            re.compile('^.*完成(.*)人民币-(' + money_re + ').*?(余额(' + money_re + ')元?)?。$'),
            [1, 2, 5]
        ]
    ],
    '[招商银行]': [
        [
            re.compile('^.*(入账|代收|转入)(.*)人民币(' + money_re + ').*(you_can_not_match)?$'),
            [1, 2, 4]
        ],
        [
            re.compile('^.*完成(.*)人民币-(' + money_re + ').*?(余额(' + money_re + ')元?)?。$'),
            [1, 2, 5]
        ]
    ],
}


def get_ana_re(content):
    for k in tempate_dict.keys():
        if not content.startswith(k) and not content.endswith(k):
            continue
        result = [k]
        templates = tempate_dict[k]
        if len(templates[0]) != 0:
            get_re = re.match(templates[0][0], content)
            if get_re is not None:
                result.append(1)
                for i in templates[0][1]:
                    result.append(get_re.group(i))
                if result[2] is not None and result[2].__contains__('工资'):
                    result[1] = 2
                return result
        if len(templates[1]) != 0:
            pay_re = re.match(templates[1][0], content)
            if pay_re is not None:
                result.append(0)
                for i in templates[1][1]:
                    result.append(pay_re.group(i))
                return result
        print('无法匹配模板:', content)
        return None


template = re.compile(
    '^.*收入([(（](.*)[)）])?(' + money_re + ').*余额(' + money_re + ').*$')
re.match(template,
         '您尾号0304卡6月14日20:09网上银行收入(财付通转入 DF)1,000元，余额-517.81元，可用余额2,481.19元。工银信用卡【工商银行】').groups()

get_ana_re('您的账户9363，于06月16日支取人民币933.87元，交易后余额573.79【中国银行】')

test = [line.strip() for line in open('test.txt', encoding='utf8')]

[get_ana_re(l) for l in test]

'''
模板分析，区分好坏之间的差异
'''
'''
交易频率：按小时统计，统计笔数和金额
'''
df = pd.read_csv('d:/resources/sms_analysis/df_use.csv', index_col=0).drop_duplicates()
df_usr_ana = df[['mem_id', 'bad']].drop_duplicates()
duplicates_mem_id = df_usr_ana.groupby('mem_id').count()
duplicates_mem_id = duplicates_mem_id[duplicates_mem_id['bad'] > 1].index

df_usr_ana.set_index('mem_id', inplace=True)
df_usr_ana.loc[duplicates_mem_id, 'bad'] = 1
df_usr_ana['mem_id'] = df_usr_ana.index
df_usr_ana.drop_duplicates(inplace=True)
df_usr_ana.set_index([np.arange(df_usr_ana.shape[0])], inplace=True)

df_bad = df_usr_ana[df_usr_ana['bad'] == 1]


def mem_analysis(cols):
    total_fre = cols.shape[0]
    total_fee = cols['fee'].sum()
    total_ave = total_fee / total_fre

    cols_hour = cols.groupby(by=['month', 'day', 'hour'])
    cols_hour_fre = cols_hour.size()
    cols_hour_fee = cols_hour['fee'].sum()
    hour_max_fre = cols_hour_fre.max()
    hour_max_fee = cols_hour_fee.max()
    hour_max_fre_index = cols_hour_fre[cols_hour_fre == hour_max_fre].index
    hour_fremax_avefee = (cols_hour_fee.loc[hour_max_fre_index] / hour_max_fre)[0]

    cols_day = cols.groupby(by=['month', 'day'])
    cols_day_fre = cols_day.size()
    cols_day_fee = cols_day['fee'].sum()
    day_max_fre = cols_day_fre.max()
    day_max_fee = cols_day_fee.max()
    day_max_fre_index = cols_day_fre[cols_day_fre == day_max_fre].index
    day_fremax_avefee = (cols_day_fee.loc[day_max_fre_index] / day_max_fre)[0]

    total_days = cols_day.count().shape[0]

    return pd.DataFrame({'total_days': [total_days],
                         'total_fre': [total_fre],
                         'total_fee': [total_fee],
                         'total_ave': [total_ave],
                         'hour_max_fre': [hour_max_fre],
                         'hour_max_fee': [hour_max_fee],
                         'hour_ave_of_max_fre': [hour_fremax_avefee],
                         'day_max_fre': [day_max_fre],
                         'day_max_fee': [day_max_fee],
                         'day_ave_of_max_fre': [day_fremax_avefee]})


df_handle = df.groupby('mem_id').apply(mem_analysis)
df_handle['mem_id'] = df_handle.index.levels[0]

df_ana = pd.merge(df_handle, df_usr_ana, how='left', on='mem_id')

df_ana_bad = df_ana[df_ana['bad'] == 1]
df_ana_good = df_ana[df_ana['bad'] != 1]

import seaborn as sns
import matplotlib.pyplot as plt

sns.factorplot('first_bad', 'hour_max_fee', data=df_ana)


def plot_bad_ratio(total, bad, col, up=True):
    va = total[col]
    min_ = va.min()
    max_ = va.max()
    x = np.linspace(min_, max_, 500)[1:-1]
    if up:
        y = [bad[bad[col] > t].shape[0] / total[total[col] > t].shape[0] for t in x]
    else:
        y = [bad[bad[col] < t].shape[0] / total[total[col] < t].shape[0] for t in x]
    plt.plot(x, y)


g = sns.FacetGrid(df_ana, hue='bad', aspect=3)
g.map(sns.kdeplot, 'day_max_fre', shade=True).add_legend()

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler


def get_best_para(x, y, c_max=1.4, cv=4, metric_n='roc_auc'):
    estimator = LogisticRegression(class_weight='balanced', penalty='l2')
    param_grid = {'solver': ['newton-cg', 'lbfgs', 'liblinear'], 'C': np.arange(0.1, c_max, 0.05), \
                  'fit_intercept': [True, False], 'intercept_scaling': [0.5, 1., 1.2, 1.5, 2]}
    clf = GridSearchCV(estimator=estimator, scoring=metric_n, param_grid=param_grid, n_jobs=-1,
                       cv=cv)
    clf.fit(x, y)
    return clf.best_estimator_, clf.best_params_, clf.best_score_


x = df_ana.loc[:, 'day_ave_of_max_fre':'total_fre']
y = df_ana['bad']

sc = StandardScaler().fit(x)
x_norm = sc.transform(x)

be, bp, bs = get_best_para(x_norm, y)
