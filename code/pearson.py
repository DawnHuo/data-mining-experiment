import numpy as np
from scipy.stats import pearsonr
import math

# def cal_pearson():



path = '../data/trainingData.txt'
data = np.loadtxt(path, dtype=int, delimiter=',')
path = '../data/pearson.txt'
fw = open(path, 'w')

d = dict()
for line in data:
    uid = line[0]
    itemid = line[1]
    rate = line[2]
    flag = d.get(itemid, -1)
    if flag == -1:
        d[itemid] = [[uid], [rate]]
    else:
        users = flag[0]
        rates = flag[1]
        users.append(uid)
        rates.append(rate)
        d[itemid] = [users, rates]

# caculate pearson correlation coefficient
pearson = []
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
            s = str(key1) + ',' + str(key2) + ',' + '0'
            pearson.append(s)
        else:
            s = str(key1) + ',' + str(key2) + ',' + str(p[0])
            pearson.append(s)

for p in pearson:
    s = str(p) + "\n"
    fw.write(s)
fw.close()
