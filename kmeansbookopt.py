# 分别导入numpy、matplotlib以及pandas，用于数学运算吧、作图和数据分析
import numpy as np
import pandas as pd
# 使用ARI进行K-means聚类性能评估
# 从sklearn导入度量函数库metrics
from sklearn.cluster import KMeans
from sklearn import metrics

# 使用pandas分别读取训练数据与测试数据
digits_train = pd.read_csv('optdigits.tra', header=None)
digits_test = pd.read_csv('optdigits.tes', header=None)

# 从训练与测试数据集上都分离出64维的像素特征与一维度的数字目标
X_train = digits_train[np.arange(64)]
Y_train = digits_train[64]
X_test = digits_test[np.arange(64)]
Y_test = digits_test[64]

# 初始化Kmeans模型，并设置聚类中心的数量为10
kmeans = KMeans(n_clusters=10)
kmeans.fit(X_train)

# #逐条判断每个测试图像所属的聚类中心
# Y_pred=kmeans.predict(X_test)
#
# #使用ARI进行k-means聚类性能评估
# print (metrics.adjusted_rand_score(Y_test,Y_pred))

Y_pred = kmeans.predict(X_train)
print(metrics.adjusted_rand_score(Y_train, Y_pred))
