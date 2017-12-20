import numpy as np
import pandas as pd
import math
from sklearn.neighbors import NearestNeighbors


def hot_analysis(data, distance_func, eps, thresh_hold):
    """
    用来分析
    :param data:数据，numpy数组,n*2,依次为经度、纬度
    :param distance_func:距离函数
    :param eps: 半径
    :param thresh_hold: 阈值（最小样本数）
    :return:
        一个3维数组：
        0：热点index
        1：对应的周边点的数量
        3：所有周边临近点的索引
    """
    neighbors_model = NearestNeighbors(radius=eps, algorithm='auto',
                                       metric=distance_func)
    neighbors_model.fit(data)
    # This has worst case O(n^2) memory complexity
    neighborhoods = neighbors_model.radius_neighbors(data, eps, return_distance=True)
    dis_mean = np.array([np.mean(i) for i in neighborhoods[0]])
    ncounts = np.array([len(i) for i in neighborhoods[1]])
    ncounts_series = pd.Series(data=ncounts)
    ncounts_valid = ncounts_series[ncounts_series > thresh_hold]
    indexes = list(ncounts_valid.index.values)
    result_indexes = []
    # 一簇只返回最中心的一个值
    while len(indexes) != 0:
        tem_index = -1
        for index in indexes:
            if tem_index == -1 or ncounts[index] > ncounts[tem_index] \
                    or (ncounts[index] == ncounts[tem_index]
                        and dis_mean[index] < dis_mean[tem_index]):
                tem_index = index
        result_indexes.append(tem_index)
        if tem_index in indexes:
            indexes.remove(tem_index)
        for index_to_del in neighborhoods[1][tem_index]:
            if index_to_del in indexes:
                indexes.remove(index_to_del)
    result = np.array([result_indexes, ncounts[result_indexes], neighborhoods[1][result_indexes]])
    return result


def rad(angle):
    return angle * math.pi / 180


def lon_lat_distance_obj(loc1, loc2):
    return lon_lat_distance(loc1[0], loc1[1], loc2[0], loc2[1])


def lon_lat_distance(lon1, lat1, lon2, lat2):
    """
    计算经纬度距离
    :param lon1:第一个地址的经度
    :param lat1:第一个地址的纬度
    :param lon2:第二个地址的经度
    :param lat2:第二个地址的纬度
    :return:
    """
    if abs(lat1) > 90 or abs(lat2) > 90 or abs(lon1) > 180 or abs(lon2) > 180:
        return -1
    rad_lat1 = rad(lat1)
    rad_lat2 = rad(lat2)
    a = rad(lat1 - lat2)
    b = rad(lon1 - lon2)
    s = 2 * math.asin(math.sqrt(pow(math.sin(a / 2), 2) + math.cos(rad_lat1) * math.cos(
        rad_lat2) * math.pow(math.sin(b / 2), 2)))
    s = s * 6378.137 * 1000
    return s


def lon_lat_similarity(lon1, lat1, lon2, lat2):
    dis = lon_lat_distance(lon1, lat1, lon2, lat2) / 1000  # 按千米为单位
    return pow(math.e, -dis)


from workspace.address_ana.edit_distance import *


def address_similarity(dict1, dict2):
    """
    计算两个地址的相似度
    :param dict1: 地址一的信息dict1['address']：地址 dict1['lon']：经度 dict1['lat']纬度
    :param dict2: 地址二的信息，类同dict1
    :return: 相似度，0-1，1表示完全相似
    """
    if type(dict1) != dict:
        dict1 = {'address': dict1[0], 'lon': dict1[1], 'lat': dict1[2]}
    if type(dict2) != dict:
        dict2 = {'address': dict2[0], 'lon': dict2[1], 'lat': dict2[2]}

    es = edit_similarity(dict1['address'], dict2['address'])
    ls = lon_lat_similarity(dict1['lon'], dict1['lat'], dict2['lon'], dict2['lat'])
    print(es, ls)

    return ls * (0.2 * es + 0.8)


df = pd.read_csv('d:/tem.csv', encoding='gb18030', index_col=0, dtype='str')
df_add_loc = df[['自填地址', 'good_bad', 'location', 'add_time']].dropna()
lon_lat = df_add_loc['location'].apply(
    lambda x: [float(x.split(',')[0]), float(x.split(',')[1])])

lon_lat_np = np.array(list(lon_lat.values))

similarity_data = pd.DataFrame()
similarity_data['address'] = df_add_loc['自填地址']
similarity_data['lon'] = lon_lat_np[:, 0]
similarity_data['lat'] = lon_lat_np[:, 1]
similarity_data['good_bad'] = df_add_loc['good_bad']
similarity_data['add_time'] = df_add_loc['add_time']

similarity_data_bad = similarity_data[similarity_data['good_bad'] == '1.0']

similarity_data_np = similarity_data.values

similarity_data_bad_np = similarity_data_bad.values

hot_re = hot_analysis(similarity_data_np, address_similarity, 0.5, 5)

hot_re1 = hot_analysis(similarity_data_np[:, 1:3], lon_lat_distance_obj, 200, 5)

hot_bad = hot_analysis(similarity_data_bad_np[:, 1:3], lon_lat_distance_obj, 200, 5)


def bad_radio(df):
    bad_num = df[df['good_bad'] == '1.0'].shape[0]
    total_num = df.shape[0]
    return bad_num / total_num


bad_radioes = []
for i in range(len(hot_re1[0])):
    radio = bad_radio(similarity_data.iloc[hot_re1[2][i]])
    bad_radioes.append(radio)

bad_ratio_np = np.array(bad_radioes)

bad_ratio_pd = pd.Series(bad_ratio_np)


def getIndex(thresh_hold):
    return bad_ratio_pd[bad_ratio_pd > thresh_hold].sort_values(ascending=False).index.values


def show(i):
    print(similarity_data.iloc[hot_re1[1][i]][['lon', 'lat']])
    print(similarity_data.iloc[hot_re1[2][i]].sort_values('add_time')[['add_time', 'address',
                                                                       'good_bad']])
    print(bad_ratio_np[i])
