import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import jieba
from urllib import request
from urllib.parse import quote
import re
import json

'''
df = pd.read_csv('D:/resources/address_ana/address.csv', header=None)
df.columns = ['province', 'city', 'district', 'address', 'date']
清理在地址中出现的省、市、区
def clear_duplicated_address(row):
    province = row['province']
    city = row['city']
    district = row['district']
    address = row['address']
    return re.sub(province + "|" + city + "|" + district, "", address)
df_clear_address = df.apply(clear_duplicated_address, axis=1)
df.address = df_clear_address
'''
df = pd.read_csv('D:/resources/address_ana/address_clear_pc.csv')
df.fillna('Unknown', inplace=True)
df['address'] = df['address'].apply(lambda x: re.sub('(\\\\\\\\n|\n|\r)', "", x))

df_gb = df.groupby(by=['province', 'city', 'district'])

key = '44ce2d09304c769ca097875fece5c34b'
p_url = "http://restapi.amap.com/v3/geocode/geo?key=" + key + "&address=%s&city=%s"
r_url = "http://restapi.amap.com/v3/geocode/regeo?key=" + key + "&location=%s"


def gaode_api(url, *para):
    act_url = url % para
    response = request.urlopen(quote(act_url, safe='/:?=&'))
    return json.loads(response.readline().decode())


# 取出1000条数据用作测试
test_data = df[df['city'] == '重庆市'].copy()

'''
调用高德api获取经纬度和详细地址
'''

def api_handle(row):
    city = row['city']
    address = row['address']
    encode = gaode_api(p_url, address, city)
    if len(encode['geocodes']) != 0:
        location = encode['geocodes'][0]['location']
        decode = gaode_api(r_url, location)
        after_format = decode['regeocode']['formatted_address']
    else:
        location = None
        after_format = None
    return [location, after_format]


test_date_after_api = test_data.apply(api_handle, axis=1)

test_date_after_api_np_array = np.array(list(test_date_after_api.values))

test_data['location'] = test_date_after_api_np_array[:, 0]
test_data['gaode_format'] = test_date_after_api_np_array[:, 1]


def clear_duplicated_address(row, col):
    province = row['province']
    city = row['city']
    district = row['district']
    address = row[col]
    if address is None or address == "None":
        return None
    return re.sub(province + "|" + city + "|" + district, "", address)


test_data['gaode_format'] = test_data.apply(clear_duplicated_address, axis=1, col='gaode_format')
test_data['address'] = test_data.apply(clear_duplicated_address, axis=1, col='address')



number_map = ['零', '一', '二', '三', '四', '五', '六', '七', '八', '九']
number_map1 = ['零', '壹', '贰', '叁', '肆', '伍', '陆', '柒', '捌''玖', '拾']


def clear_upper_number(row):
    r = list(row)
    for i in range(len(r)):
        if r[i] in number_map:
            r[i] = str(number_map.index(r[i]))
            continue
        if r[i] in number_map1:
            r[i] = str(number_map1.index(r[i]))
    return ''.join(r)


test_data['address'] = test_data['address'].apply(lambda x: re.sub('(\\\\\\\\n|\n|\r)', "", x))
test_data['address'] = test_data['address'].apply(clear_upper_number)
test_data.to_csv('d:/resources/address_ana/chongqing.csv', encoding='utf8', index=None)

admi_addr = []

test_data = pd.read_csv('d:/resources/address_ana/dongguan.csv', dtype='str')
test_data.fillna('unknown', inplace=True)


def edit_distance(str1, str2):
    dp = [i for i in range(len(str2) + 1)]
    saved = None
    for i in range(1, len(str1) + 1):
        for j in range(len(str2) + 1):
            if j == 0:
                saved = dp[j]
                dp[j] = i
                continue
            v1 = saved
            saved = v2 = dp[j]
            v2 += 1
            v3 = dp[j - 1] + 1
            if str1[i - 1] != str2[j - 1]:
                v1 += 1
            dp[j] = min(v1, v2, v3)
    return dp[len(str2)]


dongguan_address = list(test_data[test_data.district == '东莞市'].address.values)
distance_metrics = []
for i in range(len(dongguan_address)):
    print(i)
    tem = [None] * len(dongguan_address)
    for j in range(len(dongguan_address)):
        tem[j] = edit_distance(dongguan_address[i], dongguan_address[j])
    distance_metrics.append(tem)

# 距离度量的可视化展示
from sklearn.manifold import MDS
import mpl_toolkits.mplot3d
import numpy as np
import matplotlib.pyplot as plt

metrics_np = np.array(distance_metrics)
mds = MDS(n_components=3, dissimilarity='precomputed')
mds_re = mds.fit_transform(metrics_np)

ax = plt.subplot(111, projection='3d')
ax.scatter(mds_re[:, 0], mds_re[:, 1], mds_re[:, 2], label=dongguan_address)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

pattList = [['省', '特别行政区', '自治区', '市'],
            ['市', '盟', '州'],
            ['区', '县', '旗', '市', '州', '林区', '新区'],
            ['公所', '镇', '乡', '苏木', '办事处', '居委会', '社区', '街道'],
            ['村', '组', '队', '里', '园', '庄', '弄', '舍', '头', '桥', '口', '田', '农场', '沟',
             '屯', '坡', '荡', '佃', '堡', '洼', '旗', '庄', '套', '垛', '町', '甸', '冈', '河', '店',
             '岛', '集', '坊', '庄', '路', '大道', '道', '街', '巷', '胡同', '条', '里', '村委会'],
            ['大厦', '广场', '饭店', '中心', '大楼', '场', '馆', '酒店', '宾馆', '市场', '花园',
             '招待所', '中心', '大学', '厂', '局', '宿舍'],
            ['组', '队', '园', '弄', '舍', '桥', '口', '田', '沟', '坡', '荡', '佃', '洼', '套', '垛',
             '町', '冈', '巷', '胡同', '村委会', '号', '馆', '趟', '居', '寓', '苑', '墅', '小区',
             '公寓', '号院', '花园', '大厦', '广场', '中心'],
            ['单元', '层', '室', '栋', '号楼', '幢', '座', '楼', '斋', '房间']
            ]


def handle_address(address):
    result = [None] * 9
    current_index = 3
    while current_index < len(pattList):
        current_min = -1
        split_index = len(address)
        # 优先匹配长关键字
        key_length = -1
        for key in pattList[current_index]:
            # find_list = re.findall(key, address)
            # if len(find_list) > 0 and (current_min > len(find_list[0]) or current_min == -1):
            #     current_match = find_list[0]
            # find
            tem = address.find(key)
            if tem != -1 and ((key_length == -1 or key_length < len(key)) or
                                  (current_min > tem or current_min == -1)):
                current_min = tem
                split_index = tem + len(key)
                key_length = len(key)
        if current_min != -1:
            if split_index <= 1 and result[current_index - 1] is not None:
                current_index -= 1
                result[current_index] += address[:split_index]
            else:
                result[current_index] = address[:split_index]
            address = address[split_index:]
        current_index += 1
    result[8] = address
    return result


df['address_try1'] = df['address'].apply(handle_address)

handle_address("永康南巷19号院43号楼八单元601")
