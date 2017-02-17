#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/2/15 19:04
# @Author  : GuoChang
# @Site    : https://github.com/xiphodon
# @File    : ML_demo_03.py
# @Software: PyCharm Community Edition


# 梯度下降

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from mpl_toolkits.mplot3d import Axes3D


def init():
    '''
    初始化数据
    :return:
    '''
    pga = pd.read_csv(r"data/pga.csv")
    return pga


def ML_01(data):
    '''
    梯度下降
    简单实现梯度下降
    :param data:
    :return:
    '''

    # 数据归一化处理
    data["distance"] = (data["distance"] - data["distance"].mean()) / data["distance"].std()
    data["accuracy"] = (data["accuracy"] - data["accuracy"].mean()) / data["accuracy"].std()
    # data.distance = (data.distance - data.distance.mean()) / data.distance.std()
    # data.accuracy = (data.accuracy - data.accuracy.mean()) / data.accuracy.std()
    print(data.head())

    # plt.scatter(data["distance"], data["accuracy"])
    # plt.xlabel("normalized distance")
    # plt.ylabel("normalized accuracy")
    # plt.show()

    print("shape of the series:", data["distance"].shape)
    print("shape with newaxis:", data["distance"][:, np.newaxis].shape) # 创建新的一列，多加一个维度

    lr = LinearRegression()
    lr.fit(data["distance"][:, np.newaxis], data["accuracy"])
    theta_1 = lr.coef_[0]
    print(theta_1)

    # 简单实现代价函数
    def cost(theta_0, theta_1, x, y):
        '''
        代价函数
        :param theta_0: 偏移量
        :param theta_1: 权重量
        :param x: 数据集
        :param y: 数据集对应标签
        :return: 预测代价
        '''
        J = 0
        m = len(x) # 数据长度
        for i in range(m):
            h = theta_1 * x[i] + theta_0 # 回归预测值
            J += (h - y[i]) ** 2 # 预测值与真实值差的平方，累加

        J /= (2 * m) # 平均值，即代价
        return J

    print(cost(0, 1, data["distance"], data["accuracy"]))

    theta_0 = 100
    theta_1_list = np.linspace(-3, 2, 100)
    costs = []
    for theta_1 in theta_1_list:
        costs.append(cost(theta_0, theta_1, data["distance"], data["accuracy"]))

    plt.plot(theta_1_list, costs) # 画出theta_1 与其对应的 代价值
    plt.show()


    # 画出theta_0 和theta_1与其对应的代价值(例子)
    x = np.linspace(-10, 10, 100)
    y = np.linspace(-10, 10, 100)

    # 生成网络采样点
    X, Y = np.meshgrid(x, y)

    Z = X ** 2 + Y ** 2

    fig = plt.figure()
    ax = fig.gca(projection="3d")

    ax.plot_surface(X=X, Y=Y, Z=Z)
    plt.show()


    # 简单实现theta_0 和theta_1与其对应的代价值
    def partial_cost_theta_1(theta_0, theta_1, x, y):
        '''
        theta_1 局部梯度下降最大导数
        :param theta_0:
        :param theta_1:
        :param x:
        :param y:
        :return:
        '''
        h = theta_0 + theta_1 * x
        diff = (h - y) * x
        partial = diff.sum() / (x.shape[0])
        return partial

    def partial_cost_theta_0(theta_0, theta_1, x, y):
        '''
        theta_0 局部梯度下降最大导数
        :param theta_0:
        :param theta_1:
        :param x:
        :param y:
        :return:
        '''
        h = theta_0 + theta_1 * x
        diff = h - y
        partial = diff.sum() / (x.shape[0])
        return partial

    partial_1 = partial_cost_theta_1(0, 5, data["distance"], data["accuracy"])
    print("partail_1 = ", partial_1)

    partial_0 = partial_cost_theta_0(0, 5, data["distance"], data["accuracy"])
    print("partail_0 = ", partial_0)

    def gradient_descent(x, y, alpha=0.1, theta_0=0, theta_1=0):
        '''
        梯度下降
        :param x: 数据集
        :param y: 数据集标签
        :param alpha: 学习率
        :param theta_0:
        :param theta_1:
        :return:
        '''
        max_epochs = 1000 # 最大迭代次数
        counter = 0 # 迭代次数
        c = cost(theta_0, theta_1, x, y) # 初始化代价值
        costs = [c] # 代价值列表

        convergence_thres = 0.000001 # 收敛阈值（停止条件）
        cprev = c + 10
        theta_0_list = [theta_0]
        theta_1_list = [theta_1]

        while (np.abs(cprev - c) > convergence_thres) and (counter < max_epochs):
            # 两次梯度下降差在收敛阈值内或达到最大收敛次数时，终止循环
            cprev = c
            updata_0 = alpha * partial_cost_theta_0(theta_0, theta_1, x, y)
            updata_1 = alpha * partial_cost_theta_1(theta_0, theta_1, x, y)

            theta_0 -= updata_0
            theta_1 -= updata_1

            theta_0_list.append(theta_0)
            theta_1_list.append(theta_1)

            c = cost(theta_0, theta_1, x, y)

            costs.append(c)
            counter += 1

        return {"theta_0":theta_0, "theta_1":theta_1, "costs":costs}

    print("Theta_1 = ", gradient_descent(data["distance"], data["accuracy"])["theta_1"])

    descend = gradient_descent(data["distance"], data["accuracy"], alpha=0.01)
    plt.scatter(range(len(descend["costs"])), descend["costs"]) # 横轴为迭代次数，纵轴为代价值
    plt.show()



if __name__ == "__main__":
    data = init()
    ML_01(data)