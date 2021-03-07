from __future__ import print_function

import numpy as np
import torch


def get_target_distribution_sampler(mu, sigma):
    """
    采样目标数据分布的样本

    :param mu: 均值
    :param sigma: 方差
    :return: size=(1, n) 大小的样本
    """
    # Gaussian：正态分布、高斯分布
    # np.random.normal(mu, sigmoid, size=None)
    # size默认为None, 若size为None, 则返回单个样本; 否则(m, n), 则返回: m * n 个样本
    # 返回 lambda 表达式：匿名函数, lambda 参数: 返回值
    return lambda n: torch.Tensor(np.random.normal(mu, sigma, size=(1, n)))


def get_generator_input_sampler():
    """
     初始生成器(generator)数据分布：采用均匀分布(Uniform), 而非随机噪声, 以保证 Generator必须以非线性方法生成目标数据分布

    :return: 生成size为([m, n]), 范围为(0, 1)样本
    """
    # 均匀分布
    # torch.rand(m, n)
    # lambda表达式：匿名函数, lambda 参数: 返回值
    return lambda m, n: torch.rand(m, n)