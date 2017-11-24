import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('D:\\resources\\loan-control\\trade-8-9-10m.csv', header=None)

data.columns = ['time', 'value']

data['date'] = data['time'].apply(lambda x: x.split()[0])
data['month'] = data['date'].apply(lambda x: int(x.split('-')[1]))
data['day'] = data['date'].apply(lambda x: int(x.split('-')[2]))
data['hour'] = data['time'].apply(lambda x: int(x.split()[1][0:2]))

data_gb = data.groupby(by=['month', 'day', 'hour']).agg(np.sum).reset_index()

hour_data = data_gb['value'].values
day_data = data_gb.groupby(by=['month', 'day']).value.agg('sum').values
hour_cross_day_data = []
for i in range(24):
    t = data_gb[data_gb.hour == i].value.values
    hour_cross_day_data.append(t)

