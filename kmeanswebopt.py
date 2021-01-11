from collections import defaultdict
from random import uniform
from math import sqrt
import pandas as pd
# 使用ARI进行K-means聚类性能评估
from sklearn import metrics


# 读取文件
def readdata():
    dataset = []
    with open('optdigits.tra', 'r') as file:
        for line in file:
            if line == '\n':
                continue
            # map()是 Python 内置的高阶函数，它接收一个函数 f 和一个 list，
            # 并通过把函数 f 依次作用在 list 的每个元素上，得到一个新的 list 并返回。
            dataset.append(list(map(int, line.split(',')[:-1])))
        file.close()
        return dataset


# 将结果写入文件
def write_results(listResult, k, rate):
    with open('resultwebopt.txt', 'a') as file:
        #先填写准确率
        file.write('%.16f\n' % rate)
        # 按类别书写
        for kind in range(k):
            file.write("CLASSINFO:%d\n" % (kind + 1))
            for j in listResult[kind]:
                file.write('%d\t' % j)
            file.write('\n')
        file.write('\n\n')
        file.close()


# 输入：每一类的数据
# 输出：这一类新的质心
# 方法：求出这一类数据的总和求平均值
def point_avg(points):
    dimensions = len(points[0])
    new_center = []
    for dimension in range(dimensions):
        sum = 0
        #求出每一列的数据总和
        for p in points:
            sum += p[dimension]
        new_center.append(float("%.8f" % (sum / float(len(points)))))
    return new_center


# 输入：数据集，质心数据，质心数量
# 输出：新的质心数据
# 功能：更新质心数据
def update_centers(data_set, assignments, k):
    # new_means，记录前一次学习的标签结果
    new_means = defaultdict(list)
    centers = []
    for assignment, point in zip(assignments, data_set):
        new_means[assignment].append(point)
    for i in range(k):
        # 逐一对每一类进行质心更新
        points = new_means[i]
        centers.append(point_avg(points))
    return centers


# 输入：数据集，质心
# 输出：数组，每一个单元代表对应数据的类别。如assignment[0]=0，代表第0个数据的类别是第0类
# 功能：求数据集每个单元数据的标签
def assign_points(data_points, centers):
    assignments = []
    for point in data_points:
        # float('inf')表示正无穷
        shortest = float('inf')
        shortest_index = 0
        # 取距离最近的质心的下标
        for i in range(len(centers)):
            value = distance(point, centers[i])
            if value < shortest:
                shortest = value
                shortest_index = i
        assignments.append(shortest_index)
    # set是无序不重复数据集，len(set(assignments))代表数据集的标签种类
    if len(set(assignments)) < len(centers):
        print("\n--!!!产生随机数错误，请重新运行程序！!!!--\n")
        exit()
    return assignments


# 求欧拉距离
def distance(a, b):
    dimention = len(a)
    sum = 0
    for i in range(dimention):
        sq = (a[i] - b[i]) ** 2
        sum += sq
    return sqrt(sum)


# 输入：数据集，质心个数
# 输出：质心数据
# 功能：用随机数的方法生成初始质心
def generate_k(data_set, k):
    centers = []
    dimentions = len(data_set[0])
    min_max = defaultdict(int)
    # 求出每一列的最大值和最小值
    for point in data_set:
        for i in range(dimentions):
            value = point[i]
            min_key = 'min_%d' % i
            max_key = 'max_%d' % i
            if min_key not in min_max or value < min_max[min_key]:
                min_max[min_key] = value
            if max_key not in min_max or value > min_max[max_key]:
                min_max[max_key] = value
    # 求每一个质心的数据
    for j in range(k):
        rand_point = []
        for i in range(dimentions):
            min_val = min_max['min_%d' % i]
            max_val = min_max['max_%d' % i]
            # uniform函数，生成最小值和最大值之间的一个随机数
            tmp = float("%.8f" % (uniform(min_val, max_val)))
            rand_point.append(tmp)
        centers.append(rand_point)
    return centers


def k_means(dataset, k, Y_train):
    # 求初始质心
    k_points = generate_k(dataset, k)
    # 求初始标签
    assignments = assign_points(dataset, k_points)
    print(assignments)
    old_assignments = None

    # 直到标签没变，停止
    while assignments != old_assignments:
        # 更新质心数据
        new_centers = update_centers(dataset, assignments, k)
        old_assignments = assignments
        # 更新标签
        assignments = assign_points(dataset, new_centers)
    #输出准确率
    precision_rate = metrics.adjusted_rand_score(Y_train,assignments)
    print(precision_rate)
    #将分类结果输出到文件
    listResult = [[] for i in range(k)]
    count = 0
    for i in assignments:
        listResult[i].append(count)
        count = count + 1
    write_results(listResult, k, precision_rate)


def main():
    dataset = readdata()
    # 使用pandas分别读取训练数据与测试数据
    digits_train = pd.read_csv('optdigits.tra', header=None)
    Y_train = digits_train[64]
    k_means(dataset, 10 , Y_train)


if __name__ == "__main__":
    main()
