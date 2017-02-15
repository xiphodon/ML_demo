#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/2/15 08:59
# @Author  : GuoChang
# @Site    : https://github.com/xiphodon
# @File    : ML_demo_02.py
# @Software: PyCharm Community Edition


# 使用逻辑回归分类（留学申请数据）

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn import metrics



def init():
    '''
    初始化数据
    :return:
    '''
    admissions = pd.read_csv(r"data/admissions.csv")
    print(admissions.head())
    return admissions


def ML_01(admissions):
    '''
    留学申请数据简单展示
    :param admissions: 数据集
    :return:
    '''
    # 绩点分值gpa与是否通过申请数据展示
    plt.scatter(admissions["gpa"], admissions["admit"])
    plt.show()


def ML_02(admissions):
    '''
    使用逻辑回归分类数据
    :param admissions:
    :return:
    '''
    # 实例化逻辑回归模型
    logistic = LogisticRegression()
    # 训练数据
    logistic.fit(admissions[["gpa"]], admissions["admit"])
    # 预测（predict_proba概率表示）
    pred_probs = logistic.predict_proba(admissions[["gpa"]])
    print(pred_probs)
    # 绘制通过申请的散点图（pred_probs[:,1]表示通过申请）
    plt.scatter(admissions["gpa"], pred_probs[:,1])
    plt.show()

    # 预测（predict阶跃函数表示）
    pred = logistic.predict(admissions[["gpa"]])
    print(pred)
    # 绘制是否通过申请的散点图
    plt.scatter(admissions["gpa"], pred)
    plt.show()

    # 把预测结果加入新列
    admissions["predicted_label"] = pred
    print(admissions["predicted_label"].value_counts()) # 对预测值进行统计

    # 把真实结果copy入新列
    admissions["actual_label"] = admissions["admit"]
    # 真实结果与预测结果匹配的布尔结果
    matches = admissions["actual_label"] == admissions["predicted_label"]
    correct_predictions = admissions[matches] # 获取真实结果与预测结果匹配的数据集
    print(correct_predictions.head())
    accuracy = len(correct_predictions) / float(len(admissions)) # 计算匹配集的占比，即预测正确率
    print(accuracy)


    # 模型效果衡量标准
    true_positive_filter = (admissions["predicted_label"] == 1) & (admissions["actual_label"] == 1) # 正确肯定
    true_positive = len(admissions[true_positive_filter]) # 正确肯定频数
    false_negative_filter = (admissions["predicted_label"] == 0) & (admissions["actual_label"] == 1) # 错误否定
    false_negative = len(admissions[false_negative_filter]) # 错误否定频数
    true_negative_filter = (admissions["predicted_label"] == 0) & (admissions["actual_label"] == 0)  # 正确否定
    true_negative = len(admissions[true_negative_filter])  # 正确否定频数
    false_positive_filter = (admissions["predicted_label"] == 1) & (admissions["actual_label"] == 0)  # 错误肯定
    false_positive = len(admissions[false_positive_filter])  # 错误肯定频数
    TPR = true_positive / float(true_positive + false_negative) # 真正率，对正类做出的正确判断，越高越好
    TNR = true_negative / float(true_negative + false_positive) # 真负率，对负类做出的正确判断，越高越好
    print(TPR)
    print(TNR)


def ML_03(admissions):
    '''
    ROC指标与测试集的价值
    :param admissions: 数据集
    :return:
    '''
    np.random.seed(8)
    # 把真实结果copy入新列
    admissions["actual_label"] = admissions["admit"]
    admissions = admissions.drop("admit", axis=1)

    # 打乱数据索引顺序
    shuffled_index = np.random.permutation(admissions.index)
    print(shuffled_index)

    # 按照打乱索引取出数据集
    shuffled_admissions = admissions.loc[shuffled_index]
    print(shuffled_admissions.head())

    train = shuffled_admissions.iloc[0:515] # 训练集
    _test = shuffled_admissions.iloc[515:len(shuffled_admissions)] # 测试集
    test = _test.copy()

    # 实例化逻辑回归模型
    lr = LogisticRegression()
    lr.fit(train[["gpa"]], train["actual_label"]) # 训练训练集

    labels = lr.predict(test[["gpa"]]) # 利用训练好的模型预测测试集
    test["predicted_label"] = labels # 创建预测值新列

    matches = test["predicted_label"] == test["actual_label"] # 匹配预测结果
    correct_predictions = test[matches]
    accuracy = len(correct_predictions) / float(len(test)) # 精准度
    print(accuracy)

    # 模型效果衡量标准
    true_positive_filter = (test["predicted_label"] == 1) & (test["actual_label"] == 1) # 正确肯定
    true_positive = len(test[true_positive_filter]) # 正确肯定频数
    false_negative_filter = (test["predicted_label"] == 0) & (test["actual_label"] == 1) # 错误否定
    false_negative = len(test[false_negative_filter]) # 错误否定频数
    true_negative_filter = (test["predicted_label"] == 0) & (test["actual_label"] == 0)  # 正确否定
    true_negative = len(test[true_negative_filter])  # 正确否定频数
    false_positive_filter = (test["predicted_label"] == 1) & (test["actual_label"] == 0)  # 错误肯定
    false_positive = len(test[false_positive_filter])  # 错误肯定频数
    TPR = true_positive / float(true_positive + false_negative) # 真正率，对正类做出的正确判断，越高越好
    TNR = true_negative / float(true_negative + false_positive) # 真负率，对负类做出的正确判断，越高越好
    print(TPR)
    print(TNR)

    # ROC曲线
    probabilities = lr.predict_proba(test[["gpa"]]) # 逻辑回归概率预测
    FPR, TPR, thresholds = metrics.roc_curve(test["actual_label"], probabilities[:,1]) # ROC参数:真实值，概率值
    print(thresholds)
    plt.plot(FPR, TPR) # 绘制ROC曲线，参数：x轴错误肯定率，y轴正确肯定率
    plt.show()

    # 计算ROC曲线面积（越大越好）
    auc_score = metrics.roc_auc_score(test["actual_label"], probabilities[:,1])
    print(auc_score)




if __name__ == "__main__":
    admissions = init()
    # ML_01(admissions)
    # ML_02(admissions)
    ML_03(admissions)