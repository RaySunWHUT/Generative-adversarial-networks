from __future__ import print_function

import numpy as np
import torch


def write_data(filename, data):

    with open(filename, 'w') as fw:
        for e in data:
            fw.write(str(e))
            fw.write("\n")


def read_data(filename):

    with open(filename, 'r') as fr:
        result = fr.readlines()
        fr.close()
    data = []
    for i in range(len(result)):
        # str.rstrip(): 删除字符串末尾字符
        data.append(eval(result[i].rstrip('\n')))

    return data


def get_mean_and_std(data):
    """
    返回 data的均值和标准差(默认为 "总体标准差"; axis 决定计算标准差的维度)

    :param data: 数据
    :return: data的均值和标准差
    """
    return [np.mean(data), np.std(data)]



def get_numerical_characteristics(data):
    """
    返回数据(data)的 4 个数字特征(numerical characteristics):
    1. mean：均值
    2. std：标准差
    3. skewness: 偏度
    4. kurtosis: 峰度

    :param data: 数据
    :return: 一维数据: torch.Size([4])
    """
    mean = torch.mean(data)
    diffs = data - mean
    var = torch.mean(torch.pow(diffs, 2.0))
    std = torch.pow(var, 0.5)
    z_scores = diffs / std

    # 偏度：数据分布偏斜方向、程度的度量, 是数据分布非对称程度的数字特征
    # 定义: 偏度是样本的三阶标准化矩
    skewness = torch.mean(torch.pow(z_scores, 3.0))

    # excess kurtosis, should be 0 for Gaussian
    # 峰度(kurtosis): 表征概率密度分布曲线在平均值处峰值高低的特征数
    # 若峰度(kurtosis) > 3, 峰的形状会比较尖, 会比正态分布峰陡峭
    kurtoses = torch.mean(torch.pow(z_scores, 4.0)) - 3.0

    # reshape(1, )：将常量转化为torch.Size([1])型张量(Tensor)
    final = torch.cat((mean.reshape(1, ), std.reshape(1, ), skewness.reshape(1, ), kurtoses.reshape(1, )))

    return final


def decorate_with_diffs(data, exponent, remove_raw_data=False):
    """
    L2 norm: ||x-mean||
    decorate_with_diffs 作用: 将原始数据(original data)以及 L2 norm 一起返回, 使 鉴别器 D 了解更多目标数据分布的信息

    :param data: Tensor: 张量
    :param exponent: 幂次
    :param remove_raw_data: 是否移除原始数据
    :return: torch.cat([data, diffs], dim=1), dim=0, 同型张量(Tensor)按行合并; dim=1, 同型张量(Tensor)按列合并;
    """
    # dim=0, 行; dim=1, 列; keepdim: 做 mean后, 保持原数据的维度空间, 即, 原原数据为2维, mean 后仍为2维
    mean = torch.mean(data.data, dim=1, keepdim=True)

    # 利用广播(broadcast)机制进行张量(Tensor)乘法
    mean_broadcast = torch.mul(torch.ones(data.size()), mean.tolist()[0][0])
    # data - data.mean[0]
    diffs = torch.pow(data - mean_broadcast, exponent)
    if remove_raw_data:
        return torch.cat([diffs], dim=1)
    else:
        # diffs: 返回样本数据与样本平均值的偏离程度(可以是n次方(exponent))
        # 并将样本的偏离程度信息与原始样本一同输入到神经网络中
        return torch.cat([data, diffs], dim=1)