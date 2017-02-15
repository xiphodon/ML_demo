#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/2/15 07:29
# @Author  : GuoChang
# @Site    : https://github.com/xiphodon
# @File    : ML_demo_01.py
# @Software: PyCharm Community Edition



# 线性回归分析汽车油耗

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


def init():
    '''
    数据初始化
    :return:
    '''
    # 加载数据
    columns = ["mpg", "cylinders", "displacement", "horsepower", "weight", "acceleration", "model year", "origin",
               "car name"]
    # delim_whitespace空格分隔数据，names加入列名
    cars = pd.read_table(r"data/auto-mpg.data", delim_whitespace=True, names=columns)
    return cars



def ML_01(cars):
    '''
    汽车数据简单展示
    :param cars: 数据集
    :return:
    '''

    fig = plt.figure()
    ax1 = fig.add_subplot(2,1,1)
    ax2 = fig.add_subplot(2,1,2)
    # 绘制散点图（x轴列名，y轴列名，kind=散点，绘制区域）
    cars.plot("weight", "mpg", kind="scatter", ax=ax1)
    cars.plot("acceleration", "mpg", kind="scatter", ax=ax2)
    plt.show()


def ML_02(cars):
    '''
    线性回归
    :param cars: 数据集
    :return:
    '''

    # 实例化线性回归模型
    lr = LinearRegression()
    # 训练模型
    lr.fit(cars[["weight"]][10:], cars["mpg"][10:])

    # # 预测（验证）
    # predictions = lr.predict(cars[["weight"]][:10])
    # print(predictions) # 预测结果
    # print(cars["mpg"][:10]) # 真实结果

    # 数据集对应的回归模型结果
    predictions = lr.predict(cars[["weight"]])

    # 绘制散点图，真实数据与回归模型结果对比
    plt.scatter(cars["weight"], cars["mpg"], c="red")
    plt.scatter(cars["weight"], predictions, c="blue")

    plt.show()

    # 计算均方误差(真实值，预测值)
    mse = mean_squared_error(cars["mpg"], predictions)
    print(mse)
    rmse = mse ** 0.5 # 标准差
    print(rmse)

if __name__ == "__main__":
    cars = init()
    # ML_01(cars)
    ML_02(cars)