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

def init_01():
    '''
    数据初始化
    （美国国会投票数据）
    :return:
    '''
    votes = pd.read_csv(r"data/114_congress.csv")
    # print(votes.head())
    return votes

def init_02():
    '''
    数据初始化
    （nba球员数据）
    :return:
    '''
    nba = pd.read_csv(r"data/nba_2013.csv")
    # print(nba.head())
    return nba

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

    # 训练算法(显示结果为每行对应的每条数据距每个簇的中心的距离)
    senator_distances = kmeans_model.fit_transform(votes.iloc[:,3:])
    # print(senator_distances)


    # 聚类结果标签
    labels = kmeans_model.labels_
    # 交叉表（聚类结果标签为一个维度，“party”各党派为一个维度）
    print(pd.crosstab(labels, votes["party"]))
    #如表所示
    #   party   D   I   R
    #   row_0
    #   0       41  2   0
    #   1       3   0   54

    # 筛选出簇标签为1且党派为D的那3位议员
    democratic_outliers = votes[(labels == 1) & (votes["party"] == "D")]
    print(democratic_outliers)

    # 散点图，横标为距簇0的距离，纵标为距纵标的距离，c=labels颜色（color）按照标签簇划分
    plt.scatter(x=senator_distances[:,0], y=senator_distances[:,1], c=labels)
    plt.show()

    # 计算离群点（距离值的平方和或立方和等等）
    extremism = (senator_distances ** 3).sum(axis=1) # 立方后按行求和
    votes["extremism"] = extremism # 增加离群点计算列“extremism”
    votes.sort_values("extremism", inplace=True, ascending=False) # 替换原值，按照离群点离群距离降序排列
    print(votes.head(10))


def ML_03(nba):
    '''
    聚类分析nba球员数据
    :param nba: nba球员数据
    :return:
    '''
    # 选择nba中位置为控球后卫的球员数据
    point_guards = nba[nba["pos"] == "PG"]
    # pts：总得分数，g：比赛次数，ppg：平均每场得分数
    point_guards["ppg"] = point_guards["pts"] / point_guards["g"]
    # print(point_guards[["pts", "g", "ppg"]].head())

    # tov：失误总数
    point_guards = point_guards[point_guards["tov"] != 0]
    # ast：助攻总数，atr：该球员贡献度的一种表示
    point_guards["atr"] = point_guards["ast"] / point_guards["tov"]

    plt.scatter(x=point_guards["ppg"], y=point_guards["atr"], c="y")
    plt.title("Point Guards")
    plt.xlabel("Points Per Game", fontsize=13)
    plt.ylabel("Assist Turnover Ratio", fontsize=13)
    plt.show()


    # K-means聚类
    num_clusters = 5 # 定义簇数
    kmeans = KMeans(n_clusters=num_clusters) # 分5簇
    kmeans.fit(point_guards[["ppg", "atr"]]) # 训练算法
    point_guards["cluster"] = kmeans.labels_ # 生成新列，标记簇标签

    def visualize_clusters(df, num_clusters):
        '''
        可视化各簇
        :param df: 数据集
        :param num_clusters: 分簇数
        :return:
        '''
        colors = ["b","g","r","c","m","y","k"]

        for n in range(num_clusters):
            clustered_df = df[df["cluster"] == n]
            plt.scatter(clustered_df["ppg"], clustered_df["atr"], c=colors[n])
            plt.xlabel("Points Per Game", fontsize=13)
            plt.ylabel("Assist Turnover Ratio", fontsize=13)

        plt.show()

    # 可视化各簇
    visualize_clusters(point_guards, num_clusters)



if __name__ == "__main__":
    votes = init_01()
    nba = init_02()
    # ML_01(votes)
    # ML_02(votes)
    ML_03(nba)