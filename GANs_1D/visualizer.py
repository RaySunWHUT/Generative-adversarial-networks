
from matplotlib import pyplot as plt
import numpy as np
from utils import get_mean_and_std

# tensor.data: data是Tensor的属性, 即存储 Tensor 数据部分, 而 x = tensor.data, 即, 提取tensor的数据部分赋给x, x 仍为Tensor类型
# tensor.detach()：从原有计算图中分离出一个tensor变量, 生成无梯度的纯tensor
# 当 requires_grad=True 时, tensor.detach()仍可以被 autograd()追踪求导, 而 tensor.data 不可被 autograd()追踪求导
# https://www.cnblogs.com/wupiao/articles/13323283.html
def extract(v):
    """
    提取张量(Tensor)存储在 storage 中的数据部分，并将数据存储到列表(list)中

    :param v: 张量
    :return: 将数据存储到列表(list)，并返回
    """
    # .storage(): 返回底层数据
    # .tolist(): 将 torch.Tensor 转换为 python 的 list(列表) 类型
    return v.data.storage().tolist()


def plot_mean_std(data):
    """
    绘制 生成器 G 最终生成的数据分布的均值(mean)和方差(std dev)

    :param data: 生成器 G 最终生成的数据分布
    """
    # 存储生成数据
    keys = []
    # 存储生成数据对应的迭代间隔
    epochs = []

    for i in range(len(data)):
        keys.append(data[i][0])
        epochs.append(data[i][1])

    means = []
    stds = []
    for j in range(len(keys)):
        mean, std = get_mean_and_std(keys[j])
        means.append(mean), stds.append(std)

    # 平铺画布
    fig = plt.figure()

    m = fig.add_subplot(2, 1, 1)

    # 绘制迭代-均值散点图
    m.scatter(epochs, means)
    m.grid(axis='both')
    m.set_xlabel('intervals')
    m.set_ylabel('mean')
    m.minorticks_on()
    m.set_ylim(0, max(means))
    start, end = m.get_ylim()
    m.yaxis.set_ticks(np.arange(start, end, 1))

    # 绘制迭代-方差散点图
    s = fig.add_subplot(2, 1, 2)
    s.scatter(epochs, stds)
    s.set_xlabel('intervals')
    s.set_ylabel('std dev')
    s.minorticks_on()
    s.set_ylim(0, max(stds))
    start, end = s.get_ylim()
    s.yaxis.set_ticks(np.arange(start, end, 1))

    # subplots_adjust：调整子图间距
    fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.5)
    fig.show()


def plot_generated_distribution(data_type, generated_fake_data):
    """
    绘制生成的目标样本分布

    :param data_type: 输入到 鉴别器 D 的数据类别
    :param generated_fake_data: 生成器 G 生成的 fake data
    """
    values = extract(generated_fake_data)
    print("Values: %s" % (str(values)))
    plt.hist(values, bins=50)
    plt.xlabel('Value')
    plt.ylabel('Count')
    plt.title(data_type + " Histogram of Generated Distribution")
    plt.grid(True)
    plt.show()


def plot_input_distribution(generator_input_data):
    """
    绘制输入样本分布

    :param generator_input_data: 生成器的输入数据分布
    """
    plt.plot(generator_input_data)
    plt.xlabel('Value')
    plt.ylabel('Count')
    plt.ylim(0, int(sum(generator_input_data) / 10))
    plt.title('Uniform Distribution of Inputs')
    plt.grid(True)
    plt.show()