"""
Author:Mingyang Wu
Day:05.02.2024
Abstract:
Tips:
"""

import torch
import torch.nn
import numpy as np


def softmax(x):
    # axis和keepdims是numpy函数参数，用于控制操作的维度和输出的形状。
    # keepdims参数是一个布尔值，用于指示操作后是否保留原始数组的维度
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    sm = exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    return sm


def crossentropy(predictions, targets, epsilon=1e-12):
    # epsilon 一个很小的数，用来防止数值计算中的除以零或对零取对数的问题
    # np.clip函数将预测值限制在[epsilon, 1-epsilon]的范围内
    # 通过这种方式，所有的预测值都会被限制在一个非零的范围内，从而避免了数值问题。
    predictions = np.clip(predictions, epsilon, 1.-epsilon)

    # 样本的数量
    N = predictions.shape[0]
    ce = -np.sum(targets * np.log(predictions)) / N

    return ce
