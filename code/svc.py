import numpy as np
from sklearn import svm

path = '../data/pearson.txt'
data = np.loadtxt(path, dtype=int, delimiter=',')
X, Y = np.split(data, (2,), axis=1)

# fit the model
clf = svm.SVC(C=0.8, kernel='rbf', gamma=20, decision_function_shape='ovr')
clf.fit(X, Y)
pre = clf.predict([1015, 204])
print(pre)
# i = 0
# accu = 0
# for d in pre:
#     accu = accu + abs(d - y_test[i])
#     i = i+1
# print('acu:', str(accu/(i-1)))