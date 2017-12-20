import requests
import urllib
import json
import pandas as pd
import numpy as np

url = "http://10.32.32.30:10402/rkg-web/web/queryUserToUser?mobile=18714812012&level=2"
dict_j = requests.get(url).json()

source_list = []
target_list = []
type_list = []
links = dict_j['data']['links']
for i in range(len(links)):
    source_list.append(links[i]['source'])
    target_list.append(links[i]['target'])
    type_list.append(links[i]['type'])

df_links = pd.DataFrame({'source': source_list, 'target': target_list, 'type': type_list})

nodes = dict_j['data']['nodes']
id_list = []
label_list = []
mem_id_list = []
mob_num_list = []

for i in range(len(nodes)):
    id_list.append(nodes[i]['id'])
    label_list.append(nodes[i]['label'])
    mem_id_list.append(nodes[i]['memberId'])
    mob_num_list.append(nodes[i]['mobile'])

df_nodes = pd.DataFrame(
    {'id': id_list, 'label': label_list, 'mem_id': mem_id_list, 'mob_num': mob_num_list})

df_use = pd.merge(left=df_links, right=df_nodes, left_on=['target'], right_on=['id'], how='inner')
df_use.rename(columns={'mem_id': 't_mem_id', 'label': 't_label', 'id': 't_id', 'mob_num': 't_mob'},
              inplace=True)
df_use = pd.merge(left=df_use, right=df_nodes, left_on=['source'], right_on=['id'], how='inner')
df_use.rename(columns={'mem_id': 's_mem_id', 'label': 's_label', 'id': 's_id', 'mob_num': 's_mob'},
              inplace=True)
del df_use['t_id'], df_use['s_id']

df_use['t_label'].replace(['User', 'OverdueUser'], [0, 1], inplace=True)
df_use['s_label'].replace(['User', 'OverdueUser'], [0, 1], inplace=True)
