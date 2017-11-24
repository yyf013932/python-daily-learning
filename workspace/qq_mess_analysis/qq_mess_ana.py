import re
import pandas as pd

f = open('test.txt', encoding='utf8')

lines = f.readlines()[1:]

pt = re.compile("={10,}\n")

row_pt = re.compile("^(\d{4}-\d{2}-\d{2})\s+(\d{1,2}:\d{1,2}:\d{1,2})\s+([^()\n]*)(\((\d+)\))*")

'''
存储数据的格式
{'u_name':{'receive':[(time,content)]},'send':[(time,content)]}
'''
data = {}

self_name = 'Youth.'


def is_separator(st):
    return pt.match(st) is not None


def put_content(u_name, m_name, time, content):
    if len(content.strip('[ \n]')) == 0:
        return
    if m_name == self_name:
        key = 'send'
    else:
        key = 'receive'
    if u_name not in data.keys():
        data[u_name] = {'send': [], 'receive': []}
    print(key, content.strip('[ \n]'))
    data[u_name][key].append((time, content.strip('[ \n]')))


state = 2
time = ''
g_name = ''
u_name = ''
m_name = ''
content = ''

for line in lines:
    if state == 0:
        if is_separator(line):
            state = 1
        else:
            g_name = line.strip('[ \n]')
    elif state == 1:
        if is_separator(line):
            state = 2
        else:
            u_name = line.strip('[ \n]').split(':')[1]
    elif state == 2:
        rp = row_pt.match(line)
        if is_separator(line):
            state = 0
            put_content(u_name, m_name, time, content)
            content = ""
        elif rp is not None:
            put_content(u_name, m_name, time, content)
            time = rp.group(1) + ' ' + rp.group(2)
            m_name = rp.group(3)
            content = ""
        else:
            content += line

df_data = []

for key in data.keys():
    r = data[key]['receive']
    s = data[key]['send']
    for d1 in r:
        df_data.append([key, 0, d1[0], d1[1]])
    for d2 in s:
        df_data.append([key, 1, d2[0], d2[1]])

df = pd.DataFrame(data=df_data, columns=['name', 's_r', 'time', 'content'])

df.to_csv("d:/qq_format.csv")
