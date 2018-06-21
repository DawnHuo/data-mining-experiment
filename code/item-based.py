import numpy as np

path = '../data/trainingData.txt'
fo = open(path, 'r')

# 生成1300*1182的多维矩阵，存储评分
count = 0
user_dict = dict()
item_dict = dict()
u_i = np.zeros(shape=(1300, 1182))
for line in fo.readlines():
    datas = line.split(',')
    item = int(datas[1])
    user = int(datas[0])
    rate = int(datas[2])

    u_i[int(datas[0]) - 1][int(datas[1]) - 1] = datas[2]
    flag = user_dict.get(user, -1)
    if flag == -1:
        user_dict[user] = [rate]
    else:
        flag.append(rate)
        user_dict[user] = flag

    flag = item_dict.get(item, -1)
    if flag == -1:
        item_dict[item] = [rate]
    else:
        flag.append(rate)
        item_dict[item] = flag

fo.close()

# 计算平均值
user_avg = np.zeros(shape=1300)
for key, value in user_dict.items():
    avg = 0
    for v in value:
        avg += v
    avg = avg / len(value)
    user_avg[int(key) - 1] = avg

item_avg = np.zeros(shape=1182)
for key, value in item_dict.items():
    avg = 0
    for v in value:
        avg += v
    avg = avg / len(value)
    item_avg[int(key) - 1] = avg

path = '../data/pearson.txt'
fo = open(path, 'r')

# 生成1182*1182的多维矩阵，存储对应物品之间的皮尔森相关系数
i_p = np.zeros(shape=(1182, 1182))
for line in fo.readlines():
    datas = line.split(',')
    i_p[int(datas[0]) - 1][int(datas[1]) - 1] = datas[2]

fo.close()

path = '../data/testData3.txt'
fo = open(path, 'r')

pre_rating = []
true_rating = []
for line in fo.readlines():
    datas = line.split(',')
    sum_rate = 0
    item = int(datas[1])
    user = int(datas[0])
    rate = int(datas[2])
    true_rating.append(rate)

    total_sim = 0
    for i in range(1182):
        p = i_p[item - 1][i]
        rating = u_i[user - 1][i]
        sum_rate += p * rating
        total_sim += abs(p)
    if total_sim == 0:
        result = item_avg[item - 1] * 0.5 + user_avg[user - 1] * 0.5
        # print("avg rate:", result, "true rate:", rate)
    else:
        result = sum_rate/total_sim + item_avg[item - 1]
        if result < 1:
            result = 1
        if result > 5:
            result = 5
        # print("result rate:", result, "true rate:", round(rate))
    pre_rating.append(result)

sum_abs = 0
i = 0
for r in pre_rating:
    sum_abs += abs(true_rating[i] - r)
    i += 1

mae = sum_abs / i
print('mae:', mae)

