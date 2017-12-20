from random import randint

'''
计算编辑距离
'''


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


def edit_similarity(str1, str2):
    edit_dis = edit_distance(str1, str2)
    return 1 - edit_dis / (max(len(str1), len(str2)) + 0.0)



def random_str(str_len):
    dic = '0123456789abcdefghijklmnopqrstuvwxyz'
    r = []
    for i in range(str_len):
        c = randint(0, len(dic) - 1)
        r.append(dic[c])
    return ''.join(r)


for i in range(50000):
    str_len = randint(15, 30)
    a = random_str(str_len=str_len)
    str_len = randint(15, 30)
    b = random_str(str_len=str_len)
    str_len = randint(15, 30)
    c = random_str(str_len=str_len)
    dis_ab = edit_distance(a, b)
    dis_ac = edit_distance(a, c)
    dis_bc = edit_distance(b, c)
    if dis_ab + dis_ac <= dis_bc:
        print("wrong!")
    if dis_ab + dis_bc <= dis_ac:
        print("wrong!")
    if dis_bc + dis_ac <= dis_ab:
        print("wrong!")
