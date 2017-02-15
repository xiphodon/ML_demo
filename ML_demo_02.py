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
from sklearn import model_selection



def init():
    '''
    初始化数据
    :return:
    '''
    admissions = pd.read_csv(r"data/admissions.csv")
    print(admissions.head())
    return admissions


def init2():
    '''
    数据初始化
    :return:
    '''
    # 加载数据
    columns = ["mpg", "cylinders", "displacement", "horsepower", "weight", "acceleration", "year", "origin",
               "car name"]
    # delim_whitespace空格分隔数据，names加入列名
    cars = pd.read_table(r"data/auto-mpg.data", delim_whitespace=True, names=columns)
    return cars



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


def ML_04(admissions):
    '''
    交叉验证
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

    # 按照新顺序重新设置索引
    admissions = shuffled_admissions.reset_index()
    print(admissions.head())

    # 添加新列“fold”，把数据分为5份
    admissions.ix[0:128, "fold"] = 1
    admissions.ix[129:257, "fold"] = 2
    admissions.ix[258:386, "fold"] = 3
    admissions.ix[387:514, "fold"] = 4
    admissions.ix[515:644, "fold"] = 5
    admissions["fold"] = admissions["fold"].astype("int") # fold列数据转型为int型
    print(admissions.head())
    print(admissions.tail())

    # 交叉验证
    def train_and_test(admissions):
        '''
        交叉验证实现方法
        :param admissions: 数据集
        :return: 交叉验证正确率均值
        '''
        fold_accuracies = [] # 正确率列表
        for fold in range(1,6): # 数据集分为5份
            lr = LogisticRegression()
            train = admissions[admissions["fold"] != fold] # 训练集
            _test = admissions[admissions["fold"] == fold] # 测试集
            test = _test.copy()

            lr.fit(train[["gpa"]], train["actual_label"]) # 训练模型
            labels = lr.predict(test[["gpa"]]) # 测试集预测结果
            test["predicted_label"] = labels # 创建测试集预测结果列

            matches = test["predicted_label"] == test["actual_label"]
            correct_predictions = test[matches] # 筛选出预测正确数量
            fold_accuracies.append(len(correct_predictions) / float(len(test))) # 计算正确率并加入列表
        print(fold_accuracies)
        return np.mean(fold_accuracies) # 返回正确率列表均值

    average_accuracy = train_and_test(admissions)
    print(average_accuracy)


    # sklearn库中使用交叉验证
    # 数据分块，参数：需要分成几个部分，是否打乱顺序，选用随机数种子
    kf = model_selection.KFold(n_splits=5, shuffle=True, random_state=8)
    lr = LogisticRegression()
    # 交叉验证，参数：分类模型，关注的特征，真实值，返回什么值，数据分块
    accuracies = model_selection.cross_val_score(lr, admissions[["gpa"]], admissions["actual_label"], scoring="accuracy", cv=kf)
    print(accuracies) # 交叉验证返回的正确率列表
    print(np.mean(accuracies)) # 平均正确率

    roc_auc_list = model_selection.cross_val_score(lr, admissions[["gpa"]], admissions["actual_label"], scoring="roc_auc", cv=kf)
    print(roc_auc_list)  # 交叉验证返回的ROC曲线积分面积列表
    print(np.mean(roc_auc_list))  # 平均值


def ML_05(cars):
    '''
    逻辑回归多分类
    (one vs all)
    :param cars: 数据集
    :return:
    '''

    # 预测汽车产地（三分类问题）

    # 把指定列值出现的种类提取出，设置为新的列，参数prefix="year"，新列为year开头，eg：year_1
    div_year = pd.get_dummies(cars["year"], prefix="year")
    # print(div_year.head())

    div_cyl = pd.get_dummies(cars["cylinders"], prefix="cyl")
    # print(div_cyl)

    # 把多个数据集按行（axis=1）拼接在一起
    cars = pd.concat([cars, div_year, div_cyl], axis=1)
    # print(cars.head())

    cars = cars.drop("year", axis=1)
    cars = cars.drop("cylinders", axis=1)
    print(cars.head())

    # 数据洗牌
    shuffled_rows = np.random.permutation(cars.index)
    shuffled_cars = cars.iloc[shuffled_rows]

    # 数据前70%行的数据作为训练集，后30%行的数据作为测试集
    train_row = int(cars.shape[0] * 0.7)
    train = shuffled_cars.iloc[:train_row]
    test = shuffled_cars.iloc[train_row:]

    # 标签
    unique_origins = cars["origin"].unique() # 保留数组中不同的值
    unique_origins.sort()

    # 训练模型字典集合
    models = {}
    # 筛选出“cyl”或“year”开头的特征
    features = [c for c in train.columns if c.startswith("cyl") or c.startswith("year")]

    x_train = train[features]
    x_test = test[features]

    # 训练分类模型
    for origin in unique_origins: # 类别有多少个就要做多少次二分类
        lr = LogisticRegression()
        y_train = train["origin"] == origin
        lr.fit(x_train, y_train)
        models[origin] = lr

    # 测试分类模型
    testing_probs = pd.DataFrame(columns=unique_origins) # 创建列名为unique_origins的空DataFrame
    for origin in unique_origins:
        testing_probs[origin] = models[origin].predict_proba(x_test)[:,1]

    print(testing_probs)



if __name__ == "__main__":
    admissions = init()
    cars = init2()
    # ML_01(admissions)
    # ML_02(admissions)
    # ML_03(admissions)
    # ML_04(admissions)
    ML_05(cars)