import numpy as np

path = '../data/trainingData.txt'
fo = open(path, 'r')

# 生成1300*1182的多维矩阵，存储评分
count = 0
u_i = np.zeros(shape=(1300, 1182))
for line in fo.readlines():
    datas = line.split(',')
    u_i[int(datas[0]) - 1][int(datas[1]) - 1] = datas[2]
    # avg += int(datas[2])
    count += 1

avg_list = []
for uid in range(u_i.shape[0]):
    total = 0
    count = 0
    for item in range(u_i.shape[1]):
        if u_i[uid][item] == 0:
            continue
        total += u_i[uid][item]
        count += 1
    avg = total / count
    avg_list.append(avg)
# print('avg:', avg_list)
fo.close()

path = '../data/pearson.txt'
fo = open(path, 'r')

# 生成1182*1182的多维矩阵，存储对应物品之间的皮尔森相关系数
i_p = np.zeros(shape=(1182, 1182))
for line in fo.readlines():
    datas = line.split(',')
    i_p[int(datas[0]) - 1][int(datas[1]) - 1] = datas[2]

fo.close()

path = '../data/testData5.txt'
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
        if p <= 0:
            continue
        sum_rate += p * rating
        total_sim += rating
    # print("sum rate:", sum_rate, " total rate:", total_sim)
    if total_sim == 0:
        result = avg_list[user - 1]/5
        # result = 0
        # sum_rate = avg
    else:
        result = sum_rate/total_sim
    pre_rating.append(result)

sum_abs = 0
i = 0
# MIN = min(pred_rating)
# MAX = max(pred_rating)
for r in pre_rating:
    # r = (r - MIN) / (MAX - MIN)
    # print("predict:", r * 5)
    # print('predict rate:', r * 5, 'true rate:', true_rating[i])
    sum_abs += abs(true_rating[i] - r * 5)
    i += 1

mae = sum_abs / i
# print("sum_abs:", sum_abs)
print('mae:', mae)