import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import jieba

name = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10']

''' 提取部分列
for s in name:
    print('line ', s)
    df = pd.read_csv('d:/behavior20_' + s + '.csv', sep='\t', low_memory=False, header=None,
                     dtype='str', error_bad_lines=False)
    act_data = df[[14, 1, 3, 9, 10, 11, 12, 15, 16, 21, 22, 23, 24, 25, 26, 27, 28, 29, 34, 43]]
    act_data.columns = ['phone', 'event_type', 'app_source', 'client_ip', 'device_type',
                        'system_no', 'device_id', 'login_typ', 'pwd_verify_type', 'home_province',
                        'home_city', 'home_district', 'home_address', 'locate_province',
                        'locate_city', 'locate_district', 'emergency_contact_mobile',
                        'emergency_contact_mobile2nd', 'result', 'add_time']
    act_data.to_csv('d:/resources/behavior_ana/' + s + '.csv', index=None,
                    encoding='utf8')
'''

total_data = pd.DataFrame(columns=['phone', 'event_type', 'app_source', 'client_ip', 'device_type',
                                   'system_no', 'device_id', 'login_typ', 'pwd_verify_type',
                                   'home_province',
                                   'home_city', 'home_district', 'home_address', 'locate_province',
                                   'locate_city', 'locate_district', 'emergency_contact_mobile',
                                   'emergency_contact_mobile2nd', 'result', 'add_time'])
for s in name:
    print('line ', s)
    df = pd.read_csv('d:/resources/behavior_ana/' + s + '.csv',
                     dtype={'phone': 'str', 'emergency_contact_mobile': 'str',
                            'emergency_contact_mobile2nd': 'str'})
    print(df.count())
    total_data = pd.concat([total_data, df])
