#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/3/7 14:27
# @Author  : GuoChang
# @Site    : https://github.com/xiphodon
# @File    : ML_demo_05.py
# @Software: PyCharm Community Edition


# 无监督学习


import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import math

def init():
    '''
    数据初始化
    （美国国会投票数据）
    :return:
    '''
    votes = pd.read_csv(r"data/114_congress.csv")
    # print(data.head())
    return votes

def ML_01(votes):
    '''
    聚类初试
    :param votes: 投票数据
    :return:
    '''
    # 统计该列各个元素出现次数
    print(votes["party"].value_counts())
    # 统计每列均值
    print(votes.mean())

    # 欧氏距离（第一行数据的[3:]列，重构形状为一行的行向量（列是-1：自动确定列数），与同样操作的第二行数据计算欧式距离计算）
    print(euclidean_distances(votes.iloc[0,3:].values.reshape(1,-1),votes.iloc[1,3:].values.reshape(1,-1)))
    # 同上，第一行与第三行的欧氏距离
    print(euclidean_distances(votes.iloc[0,3:].values.reshape(1,-1),votes.iloc[2,3:].values.reshape(1,-1)))


def ML_02(votes):
    '''
    K均值聚类初试
    :param votes:
    :return:
    '''
    # 参数 n_clusters=2：分为两簇
    kmeans_model = KMeans(n_clusters=2, random_state=1)
    # 训练算法
    senator_distances = kmeans_model.fit_transform(votes.iloc[:,3:])
    # 显示结果为每行对应的每条数据距每个簇的中心的距离
    print(senator_distances)



if __name__ == "__main__":
    votes = init()
    # ML_01(votes)
    ML_02(votes)