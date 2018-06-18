import numpy as np

path = '../data/trainingData.txt';
data = np.loadtxt(path, dtype=int, delimiter=',')

d = dict()
for line in data:
    uid = line[0]
    itemid = line[1]
    rank = line[2]
    flag = d.get(itemid, -1)
    if flag == -1:
        d[itemid] = [[uid], [rank]]
    else:
        users = flag[0]
        ranks = flag[1]
        users.append(uid)
        ranks.append(rank)
        d[itemid] = [users, ranks]

pearsonr = []
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
            i = i +1
        if len(rlist1) == 0:
            pearsonr.append([key1, key2, 0])
        else :
            pearsonr.append([key1, key2, np.corrcoef(rlist1, rlist2)])

print(pearsonr)