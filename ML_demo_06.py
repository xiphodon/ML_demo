#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/3/8 19:05
# @Author  : GuoChang
# @Site    : https://github.com/xiphodon
# @File    : ML_demo_06.py
# @Software: PyCharm Community Edition

# 预测是否放款

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import math

def init():
    '''
    数据初始化
    （放款数据）
    :return:
    '''
    loans = pd.read_csv(r"data/cleaned_loans2007.csv")
    # print(loans.head())
    # print(loans.info())

    # 约四万条数据，取小部分以便快速出结果
    return loans[:500]


def ML_01(loans):
    '''
    逻辑回归分类数据
    :param loans:
    :return:
    '''
    # 获得数据所有列名
    cols = loans.columns
    # 获取训练集列名
    train_cols = cols.drop("loan_status")
    # 获取训练集
    features = loans[train_cols]
    # 获取目标集
    target = loans["loan_status"]

    # 权重
    penalty = {
        0:5,
        1:1
    }

    # 逻辑回归，通过权重来解决样本不均衡问题，可选用默认参数，也可自定义权重比
    # lr = LogisticRegression(class_weight="balanced")
    lr = LogisticRegression(class_weight=penalty)
    # 交叉验证
    kf = KFold(features.shape[0], random_state=1)
    predictions = cross_val_predict(lr, features, target, cv=kf)
    # 生成Series类型
    predictions = pd.Series(predictions)

    # # 随机森林方法
    # rf = RandomForestClassifier(n_estimators=10, class_weight=penalty, random_state=1)
    # # 交叉验证
    # kf = KFold(features.shape[0], random_state=1)
    # predictions = cross_val_predict(rf, features, target, cv=kf)
    # # 生成Series类型
    # predictions = pd.Series(predictions)


    # 错误肯定
    FP_filter = (predictions == 1) & (loans["loan_status"] == 0)
    FP = len(predictions[FP_filter])

    # 正确肯定
    TP_filter = (predictions == 1) & (loans["loan_status"] == 1)
    TP = len(predictions[TP_filter])

    # 错误否定
    FN_filter = (predictions == 0) & (loans["loan_status"] == 1)
    FN = len(predictions[FN_filter])

    # 正确否定
    TN_filter = (predictions == 0) & (loans["loan_status"] == 0)
    TN = len(predictions[TN_filter])

    TPR = TP / (TP + FN) # 正确肯定率（根据实际业务，越大越好）
    FPR = FP / (FP + TN) # 错误肯定率（根据实际业务，越小越好）

    print(TPR)
    print(FPR)





if __name__ == "__main__":
    loans = init()
    ML_01(loans)