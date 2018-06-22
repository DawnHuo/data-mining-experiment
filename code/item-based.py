import numpy as np
from scipy.stats import pearsonr
import math
import matplotlib.pyplot as plt

# 生成1300*1182的二维表，存储评分
def ui_table():
    path = '../data/trainingData.txt'
    fo = open(path, 'r')
    udict = dict()
    idict = dict()
    u_i = np.zeros(shape=(1300, 1182))
    for line in fo.readlines():
        datas = line.split(',')
        item = int(datas[1])
        user = int(datas[0])
        rate = int(datas[2])

        u_i[int(datas[0]) - 1][int(datas[1]) - 1] = datas[2]
        flag = udict.get(user, -1)
        if flag == -1:
            udict[user] = [rate]
        else:
            flag.append(rate)
            udict[user] = flag

        flag = idict.get(item, -1)
        if flag == -1:
            idict[item] = [[user], [rate]]
        else:
            users = flag[0]
            rates = flag[1]
            users.append(user)
            rates.append(rate)
            idict[item] = [users, rates]
    fo.close()

    return u_i, udict, idict

def cal_avg(u_i, obj):
    if obj == 'u':
        result = np.zeros(shape=len(u_i))
        for i in range(len(u_i)):
            sum = 0
            count = 0
            for j in range(len(u_i[0])):
                if u_i[i][j] == 0:
                    continue
                sum += u_i[i][j]
                count += 1
            avg = sum/count
            result[i] = avg
    else:
        result = np.zeros(shape=len(u_i[0]))
        for i in range(len(u_i[0])):
            sum = 0
            count = 0
            for j in range(len(u_i)):
                if u_i[j][i] == 0:
                    continue
                sum += u_i[j][i]
                count += 1
            avg = sum / count
            result[i] = avg
    return result

# caculate pearson correlation coefficient
def cal_pearson(d):
    pearson = np.zeros(shape=(1182, 1182))
    for key1, value1 in d.items():
        for key2, value2 in d.items():
            if key1 == key2:
                continue
            rlist1 = []
            rlist2 = []
            i = 0
            for uid1 in value1[0]:
                j = 0
                for uid2 in value2[0]:
                    if uid1 == uid2:
                        rlist1.append(value1[1][i])
                        rlist2.append(value2[1][j])
                    j = j + 1
                i = i + 1
            p = pearsonr(rlist1, rlist2)
            if math.isnan(p[0]):
                pearson[int(key1) - 1][int(key2) - 1] = 0
            else:
                pearson[int(key1) - 1][int(key2) - 1] = p[0]
    return pearson

def predict(fp, i_p, u_i, item_avg):
    fo = open(fp, 'r')

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
            result = item_avg[item - 1]
        else:
            result = sum_rate / total_sim + item_avg[item - 1]
            if result < 1:
                result = 1
            if result > 5:
                result = 5
        pre_rating.append(result)
    return pre_rating, true_rating

if __name__ == "__main__":
    user_item, user_dict, item_dict = ui_table()
    # obj = 'i'
    item_avg = cal_avg(user_item, 'i')
    item_pearson = cal_pearson(item_dict)

    mae = []
    x = []
    for i in range(1, 6):
        path = '../data/testData' + str(i) + '.txt'
        pre_rating, true_rating = predict(path, item_pearson, user_item, item_avg)
        sum_abs = 0
        j = 0
        x.append(i)
        for r in pre_rating:
            sum_abs += abs(true_rating[j] - r)
            j += 1
        mae.append(sum_abs / j)
    plt.plot(x, mae, color="r", linestyle="--", marker="*", linewidth=1.0)

    # 设置坐标轴刻度
    my_x_ticks = np.arange(1, 5, 1)
    plt.ylim((-2, 2))
    plt.xticks(my_x_ticks)
    # 设置坐标轴名称
    plt.xlabel('测试文件')
    plt.ylabel('MAE')
    # 显示图片
    plt.show()