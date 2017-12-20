import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import jieba
from urllib import request
from urllib.parse import quote
import re
import json

keys = ['146634f126f85819571215c62eaad49f', 'd5b0a7f62461dfc7b63671b176aab485',
        '96a9f498619126d38a21425b0286a3bc', '44ce2d09304c769ca097875fece5c34b',
        '1fc0613f8637f44067e7f42509cc0f9d', 'fdca12d6d60678ebe0b20a044ab97e19']

enterprise_key = '88d76d106da92185b6decacc9183e0ef'

df = pd.read_csv('d:/resources/address_ana/hunan.csv', encoding='utf-8', header=0, index_col=0)

df['search_address'] = df.apply(lambda x: (x['街道级'] + x['地标级'] + x['小区级']).replace('nann', ''),
                                axis=1)

df['delete_address'] = df.apply(lambda x: (x['栋数楼号级'] + x['最终过滤地址']).replace('nann', ''),
                                axis=1)

p_url = "http://restapi.amap.com/v3/geocode/geo?key=%s&address=%s&city=%s"
r_url = "http://restapi.amap.com/v3/geocode/regeo?key=%s&location=%s"


def gaode_api(url, *para):
    act_url = url % para
    print(act_url)
    response = request.urlopen(quote(act_url, safe='/:?=&'))
    return json.loads(response.readline().decode())


def api_handle(row, key):
    city = row['市']
    # if city.endswith('自治州'):
    #     city = row['区']
    address = row['自填地址']
    encode = None
    if ~pd.isnull(address):
        encode = gaode_api(p_url, key, address, city)
        for i_ in range(1, 7):
            if len(address) <= i_ + 3 or (encode is not None and encode['status'] != 0 and len(
                    encode['geocodes']) != 0):
                break
            encode = gaode_api(p_url, key, address[:-i_], city)
    # if encode is None or encode['status'] == 0 or len(encode['geocodes']) == 0:
    #     address = row['过滤后自填地址']
    #     encode = gaode_api(p_url, key, address, city)
    if encode is not None and encode['status'] != 0 and 'geocodes' in encode.keys() \
            and len(encode['geocodes']) != 0:
        location = encode['geocodes'][0]['location']
        level = encode['geocodes'][0]['level']
        raw_format = encode['geocodes'][0]['formatted_address']
        decode = gaode_api(r_url, key, location)
        after_format = decode['regeocode']['formatted_address']
    else:
        location = None
        after_format = None
        level = None
        raw_format = None
    return [location, level, raw_format, after_format]


# 填充数据

indexes = [0, 4139, 8278, 13798, 19318, 24832]

format_data = []

format_data_np = np.array([[]])

for i in range(4, 5):
    print('start:', str(i))
    df_tem = df.iloc[indexes[i]:indexes[i + 1]]
    tem_data = df_tem.apply(api_handle, axis=1, key=keys[i + 1])
    format_data.append(tem_data)
    format_data_np = np.vstack((format_data_np, np.array(list(tem_data.values))))

save_df = pd.DataFrame(data=format_data_np)

save_df.to_csv('d:/resources/address_ana/all.csv', header=None, index=None)
