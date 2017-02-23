#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/2/17 15:45
# @Author  : GuoChang
# @Site    : https://github.com/xiphodon
# @File    : ML_demo_04.py
# @Software: PyCharm Community Edition


# 决策树算法，随机森林算法


import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import matplotlib.pyplot as plt
import math


def init():
    '''
    初始化数据（人口普查数据）
    :return:
    '''
    columns = ["age", "workclass", "fnlwgt", "education", "education_num", "marital_status", "occupation",
               "relationship", "race", "sex", "capital_gain", "capital_loss", "hours_per_week", "native_country",
               "high_income"]

    income = pd.read_csv(r"data/income.csv", names=columns)

    for name in ["workclass", "education", "marital_status", "occupation", "relationship", "race", "sex",
                 "native_country",
                 "high_income"]:
        col = pd.Categorical(income[name])  # 使该列数据全部由字符型转换成数值型
        income[name] = col.codes  # 将该列对应编号存入原数据集中，替换原数据

    return income


def ML_01(income):
    '''
    决策树算法
    :return:
    '''

    # # 分别计算数据集标签列“high_income”中，两种收入类别出现的频率
    # prob_0 = income[income["high_income"] == 0].shape[0] / income.shape[0]
    # prob_1 = income[income["high_income"] == 1].shape[0] / income.shape[0]
    # # 先计算数据集按照标签列划分的熵值
    # income_entropy = - prob_0 * math.log(prob_0, 2) - prob_1 * math.log(prob_1, 2)
    # print(income_entropy)

    def calc_entropy(column):
        '''
        计算该列熵值
        :param column: 列（一维数组）
        :return:
        '''
        # 数值型数据列，选出唯一数据，按照升序，获得每种数据值出现的次数（一维数组）
        counts = np.bincount(column)
        # 求每种数据出现的频率（出现次数/该列数据总数）（一维数组）
        probabilities = counts / len(column)

        entropy = 0

        # 计算按照该列划分的熵值
        for prob in probabilities:
            if prob > 0:
                entropy += - prob * math.log(prob, 2)

        return entropy

    # 先计算数据集按照标签列划分的熵值
    income_entropy = calc_entropy(income["high_income"])
    print(income_entropy)

    # # 由于年龄数值种类过多，对数值进行分组
    # median_age = income["age"].median()  # 取中位数
    #
    # left_split = income[income["age"] < median_age]
    # right_split = income[income["age"] >= median_age]
    #
    # # 按照年龄列划分的信息增益
    # age_information_gain = income_entropy - (
    #         (left_split.shape[0] / income.shape[0]) * calc_entropy(left_split["high_income"]) + (
    #         right_split.shape[0] / income.shape[0]) * calc_entropy(right_split["high_income"]))
    #
    # print(age_information_gain)

    def calc_information_gain(data, split_name, target_name):
        '''
        计算信息增益
        :param data: 数据集
        :param split_name: 划分的特征名
        :param target_name: 目标列名
        :return: 信息增益
        '''
        # 计算目标列名的熵值
        original_entropy = calc_entropy(data[target_name])

        column = data[split_name] # 获取划分列数据
        median = column.median() # 获取该列中位数

        # 对于数据值过于离散的，进行分组
        left_split = income[column < median]
        right_split = income[column >= median]

        to_subtract = 0
        for subset in [left_split, right_split]:
            prob = (subset.shape[0] / data.shape[0]) # 频率
            to_subtract += prob * calc_entropy(subset[target_name])

        return original_entropy - to_subtract # 信息增益

    # print(calc_information_gain(income, "age", "high_income"))

    # 列名
    columns = ["age", "workclass", "fnlwgt", "education", "education_num", "marital_status", "occupation",
               "relationship", "race", "sex", "capital_gain", "capital_loss", "hours_per_week", "native_country"]


    def find_best_column(data, target_name, columns):
        '''
        寻找划分最好的列
        :param data: 数据集
        :param target_name: 目标名
        :param columns: 列名
        :return:
        '''
        information_gains = []

        # 计算所有列作为划分的熵值，选择信息增益最大的列名
        for col in columns:
            information_gain = calc_information_gain(income, col, "high_income")
            information_gains.append(information_gain)

        highest_gain_index = information_gains.index(max(information_gains))

        highest_gain = columns[highest_gain_index]  # 信息增益最大的列名

        return highest_gain


    income_split = find_best_column(income, "high_income", columns)
    print(income_split)



def ML_02(income):
    '''
    使用sklearn.linear决策树
    :return:
    '''


    # 列名
    columns = ["age", "workclass", "fnlwgt", "education", "education_num", "marital_status", "occupation",
               "relationship", "race", "sex", "capital_gain", "capital_loss", "hours_per_week", "native_country"]

    # 实例化决策树，参数： max_depth=7决策树深度最多为7，最小划分阈值为10（超过10则再进行划分）,
    # splitter="random"表示随机选取特征，max_features="auto"表示为最大特征数
    dtc = DecisionTreeClassifier(random_state=1, max_depth=7, min_samples_split=10, splitter="random", max_features="auto")

    # 训练决策树
    dtc.fit(income[columns][:400], income["high_income"][:400])

    # 预测值
    train_predictions = dtc.predict(income[columns][:400])
    test_predictions = dtc.predict(income[columns][400:])

    # roc值训练集与测试集对比（越大越好）
    train_auc = metrics.roc_auc_score(income["high_income"][:400], train_predictions)
    test_auc = metrics.roc_auc_score(income["high_income"][400:], test_predictions)

    print(train_auc)
    print(test_auc)



def ML_03(income):
    '''
    随机森林算法(RF)
    :param income: 数据集
    :return:
    '''
    # 列名
    columns = ["age", "workclass", "fnlwgt", "education", "education_num", "marital_status", "occupation",
               "relationship", "race", "sex", "capital_gain", "capital_loss", "hours_per_week", "native_country"]

    # n_estimators=10 随机森林中有5棵树
    rfc = RandomForestClassifier(n_estimators=10, random_state=1, min_samples_leaf=2)
    rfc.fit(income[columns][:400], income["high_income"][:400])

    predictions = rfc.predict(income[columns][400:])
    print(metrics.roc_auc_score(income["high_income"][400:], predictions))



if __name__ == "__main__":
    income = init()
    # ML_01(income)
    # ML_02(income)
    ML_03(income)
