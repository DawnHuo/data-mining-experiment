from sklearn.cluster import AffinityPropagation
from sklearn.datasets.samples_generator import make_blobs
import numpy as np
import pylab as pl
from itertools import cycle


# ## 生成的测试数据的中心点
# centers = [[1, 1], [-1, -1], [1, -1]]
# ##生成数据
# Xn, labels_true = make_blobs(n_samples=150, centers=centers, cluster_std=0.5,
#                             random_state=0)



# simi = []
# for m in Xn:
#     ##每个数字与所有数字的相似度列表，即矩阵中的一行
#     temp = []
#     for n in Xn:
#          ##采用负的欧式距离计算相似度
#         s =-np.sqrt((m[0]-n[0])**2 + (m[1]-n[1])**2)
#         temp.append(s)
#     simi.append(temp)

path = '../data/pearson.txt'
fo = open(path, 'r')
# 生成1182*1182的多维矩阵，存储对应物品之间的皮尔森相关系数
# i_p = np.zeros(shape=(1182, 1182))
sim = np.zeros(shape=(1182, 1182))
for line in fo.readlines():
    datas = line.split(',')
    # temp = [int(datas[0]), int(datas[1]), datas[2].rstrip('\n')]
    # sim.append(temp)
    # print(temp)
    sim[int(datas[0]) - 1][int(datas[1]) - 1] = datas[2]

# simi = []
# for m in range(1182):
#     temp = []
#     for n in range(1182):
#         d = i_p[m][n]
#         temp.append(d)
#     simi.append(temp)


p=-50   ##3个中心
#p = np.min(simi)  ##9个中心，
#p = np.median(simi)  ##13个中心

ap = AffinityPropagation(damping=0.5, max_iter=500, convergence_iter=30,
                         preference=p).fit(sim)
cluster_centers_indices = ap.cluster_centers_indices_

label = ap.fit_predict(sim)
for l in label:
    print(l)
# print(label)
n_clusters_ = len(cluster_centers_indices)
print('n-center:', n_clusters_)
# labels = ap.labels_

# Plot result
# pl.close('all')
# pl.figure(1)
# pl.clf()
# colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
# pl.title('Estimated number of clusters: %d' % n_clusters_)
# for k, col in zip(range(n_clusters_), colors):
#     # print("test")
#     class_members = labels == k
#     cluster_center = sim[cluster_centers_indices[k]]
#     pl.plot(sim[class_members, 0], sim[class_members, 1], col + '.')
#     pl.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
#             markeredgecolor='k', markersize=14)
#     for x in sim[class_members]:
#         pl.plot([cluster_center[0], x[0]], [cluster_center[1], x[1]], col)

# for idx in cluster_centers_indices:
#     print(sim[idx])