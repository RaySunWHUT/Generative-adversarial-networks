# __future__：使用 python 3 中的函数
# python2：print "Hello, world!"
# 在 Python2 中使用 Python3 中的一些特性都是用 from __future__ import XXX 实现
from __future__ import print_function
import torch
import torch.optim as optim
import torch.nn as nn

from utils import get_mean_and_std
from utils import decorate_with_diffs
from utils import get_numerical_characteristics
from utils import write_data
from sampler import get_target_distribution_sampler
from sampler import get_generator_input_sampler
from networks import Generator
from networks import Discriminator
from visualizer import plot_input_distribution
from visualizer import plot_generated_distribution
from visualizer import extract
from visualizer import plot_mean_std


def init(flag, remove_raw_data=False):
    """
    1. name: 输入到 鉴别器 D 的数据类别
    2. d_input_func: 对 Discriminator 的网络输入大小 input_size 进行调整
    3. preprocess: 对输入到 Discriminator 数据进行预处理(pre_process), e.g.
        3.1. get_numerical_characteristics(data): 返回数据的4个数字特征:
            1. mean：均值; 2. std：标准差; 3. skewness: 偏度; 4. kurtosis: 峰度
        3.2. decorate_with_diffs(data, exponent, remove_raw_data): 将 data 处理成 diffs; 然后选择是否与 diffs 连接(cat) 后返回

    :param flag: flag=0, 采用 decorate_with_diffs, 返回维度为 torch.Size([1, 1]);
        flag=1, 采用 get_numerical_characteristics(data), 返回维度为 torch.Size([1]);
    :param remove_raw_data: 用来标识 decorate_with_diffs 是否返回 [data, diffs]; 若 remove_raw_data=True,
        则 decorate_with_diffs 只返回 diffs; 否则, decorate_with_diffs 返回 [data, diffs]
    """
    if flag == 0:
        # 返回数据及其数据方差"偏离程度"
        (data_type, preprocess, d_input_func) = ("Data and variances",
                                                 lambda data: decorate_with_diffs(data, 2.0, remove_raw_data),
                                                 lambda x: x if remove_raw_data else x * 2)
    elif flag == 1:
        # 返回数据及其数据均值"偏离程度"
        (data_type, preprocess, d_input_func) = ("Data and diffs",
                                                 lambda data: decorate_with_diffs(data, 1.0, remove_raw_data),
                                                 lambda x: x if remove_raw_data else x * 2)
    elif flag == 2:
        # 直接使用原始数据
        (data_type, preprocess, d_input_func) = ("Raw data", lambda data: data, lambda x: x)
    elif flag == 3:
        # 仅使用数据的 4 个数字特征
        (data_type, preprocess, d_input_func) = ("Only 4 numerical characteristics",
                                                 lambda data: get_numerical_characteristics(data), lambda x: 4)
    else:
        data_type, preprocess, d_input_func = None, None, None
        print("Flag input error!\n")

    return data_type, preprocess, d_input_func


# GANs原理详解：https://www.cnblogs.com/LXP-Never/p/9706790.html
def train():

    # flag: 标识符
    flag = 3
    data_type, preprocess, d_input_func = init(flag=flag)

    print("Using data [%s]" % data_type)

    # 若GPU可用, 则使用GPU; 否则, 使用CPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 打印间隔
    print_interval = 100

    # 绘图间隔
    plot_interval = 500

    dfe, dre, ge = 0, 0, 0

    d_real_data, d_fake_data, g_fake_data = None, None, None

    # 数据参数：均值, 偏差
    data_mean = 4
    data_stddev = 1.25

    # 鉴别器：获取目标数据分布
    d_sampler = get_target_distribution_sampler(data_mean, data_stddev)
    # 生成器：获取均值分布
    g_sampler = get_generator_input_sampler()

    # 鉴别器、生成器激活函数
    generator_activation_function = torch.tanh
    discriminator_activation_function = torch.sigmoid

    # 生成器参数
    g_input_size = 1
    g_hidden_size = 5
    # 输出维度为 1
    g_output_size = 1
    g_steps = 20
    G = Generator(input_size=g_input_size,
                  hidden_size=g_hidden_size,
                  output_size=g_output_size,
                  f=generator_activation_function)

    # 鉴别器参数
    d_input_size = 500
    d_hidden_size = 10
    # 输出维度为 1
    d_output_size = 1
    d_steps = 20

    # 当使用init()的第1类数据类别时, d_input_func()会根据情况, 调整 鉴别器D 的输入大小为 鉴别器D 的初始输入大小或 2 倍
    # 当使用init()的第2类数据类别时, d_input_func()会根据情况, 调整 鉴别器D 的输入大小为 鉴别器D 的初始输入大小或 2 倍
    # 当使用init()的第3类数据类别时, d_input_func()会调整 鉴别器D 的输入大小为 鉴别器D 的初始输入大小
    # 当使用init()的第4类数据类别时, d_input_func()会调整 鉴别器D 的输入大小为 4, 即, 输入数据的4个数字特征
    D = Discriminator(input_size=d_input_func(d_input_size),
                      hidden_size=d_hidden_size,
                      output_size=d_output_size,
                      f=discriminator_activation_function)

    # 二元交叉熵(BCELoss function): 计算 输出(output) 与 目标(target) 之间的 距离error(loss)
    # loss(output, target)
    loss = nn.BCELoss()

    # d, g 学习率
    d_learning_rate = 1e-3
    g_learning_rate = 1e-3
    # SGD动量
    sgd_momentum = 0.9

    # 梯度下降：解决病态曲率同时加快搜索速度的方法(自适应方法)
    # Adam、RMSProp、Momentum: 三种主流自适应算法; Adam前景明朗, Momentum更加主流
    # Adam更易于收敛到尖锐的极小值, Momentum更可能收敛到平坦的极小值, 通常认为平坦的极小值 好于 尖锐的极小值
    d_optimizer = optim.SGD(D.parameters(), lr=d_learning_rate, momentum=sgd_momentum)
    g_optimizer = optim.SGD(G.parameters(), lr=g_learning_rate, momentum=sgd_momentum)

    # 将mini-batch大小置为discriminator的输入大小
    # 目的: 每次都能让鉴别器D看到完整的目标数据分布
    # 本文设置 d_input_size = mini-batch_size, 而非 d_input_size = 1; 因为, 如果 D 没有看到real数据的完整分布的话,
    # 那么, 则会产生 Ian Goodfellow 2016年在GANs论文中提到的 "collapse to a single mode" 问题;
    # 即, 生成器G 最终生成的样本分布均值(mean)是正确的, 但样本分布方差(std)则会非常小
    mini_batch_size = d_input_size

    # 绘制初始样本分布
    generator_input_data = g_sampler(mini_batch_size, g_input_size)
    plot_input_distribution(generator_input_data)

    d_fake_list = []

    # 若GPU可用, 将 G, D 迁移到GPU运行
    G.to(device)
    D.to(device)

    # 迭代周期
    num_epochs = 5001

    # 不断在 D, G 之间进行交替训练, 通过博弈提高 D, G 的能力
    # 每个 epoch 分别交替训练 D, G 20 steps
    for epoch in range(num_epochs):

        # 训练鉴别器(Discriminator)
        for d_index in range(d_steps):
            # 1. 分别在在real、fake数据上训练 D
            # 注：此处要清楚: D.grad是loss对parameter的导数; 而SGD则是每次更新一小步(step);
            # 即, parameter = parameter - dloss/dparameter
            # 故, parameter利用PyTorch计算图反向传播填充在parameter.grad中的梯度(grad)进行更新后, 梯度(grad)便失去意义;
            # 并且, 因为反向传播的梯度(grad)会累加, 故在进行下一次SGD更新时, 需要清楚上次填充的梯度(grad)
            # https://www.yht7.com/news/97242
            D.zero_grad()

            # 1.1: 在 real data 上训练 D
            d_real_data = d_sampler(d_input_size)

            # 将变量迁移到 GPU 上
            # 只使用 d_real_data.to(device), 变量仍在CPU上
            d_real_data = d_real_data.to(device)

            d_real_decision = D(preprocess(d_real_data))

            # 由于 d_real_decision 可能为 torch.Size([1]) 或 torch.Size([1, 1])
            # 故, 先对其进行求和(sum), 然后reshape成torch.Size([1])的张量
            # nn.BCELoss(output, target): 以 1 标识 real, 以 0 标识 fake
            d_real_error = loss(torch.sum(d_real_decision).reshape(1, ), torch.ones([1]).to(device))

            # 计算/填充梯度, 但不改变 D 的权重
            d_real_error.backward()

            # 1.2: 在 fake data 上训练 D
            # 1.2.1: 使用 Generator 生成数据
            # 按Generator的输入大小，采样mini-batch个输入
            d_gen_input = g_sampler(mini_batch_size, g_input_size)

            # 将变量迁移到 GPU 上
            d_gen_input = d_gen_input.to(device)

            # https://blog.csdn.net/qq_34218078/article/details/109591000 讲的很清楚！
            # detach(): 截断反向传播的梯度流; 将原计算图中的某个node变成不需要梯度的node, 反向传播不会从这个node向前传播
            # 此处 G 仅作为随即噪音生成器(以及迭代后有一定模仿能力的生成器), 教 D 鉴别fake data; 故, 应detach来避免在这些标签上训练 G(即, 保持 G 不动)
            # 虽然d_optimizer.step()仅优化了鉴别器 D 的参数; 但是, 由于此处仍然通过生成器 G 生成了假的数据;
            # 并且, 生成器 G 在生成假数据的过程中仍然调用了前向传播(forward); 因此, 在d_fake_error进行反向传播(backward)的过程中,
            # 仍然会对生成器G的网络参数求导, 并填充到网络参数的.grad属性中; 这样做虽然没有对G的网络参数进行更新,
            # 但在训练Discriminator进行迭代的过程中, 确实会对生成器G网络参数的 .grad 属性中, 形成梯度积累;
            # 虽然, 在迭代训练生成器 G 之前, 采用了 G.zero_grad() 进行梯度清零, 训练 鉴别器 D 时, 生成器 G 形成的梯度积累不会影响
            # 到生成器 G 的网络参数更新; 但梯度的计算开销, 内存开销以及时间成本是完全没有必要的; 故, 在训练 D 时, 采用 G.detach()
            # 截断对G的反向传播梯度流;
            d_fake_data = G(d_gen_input).detach()

            # x.t(): 对 x 进行转置
            d_fake_decision = D(preprocess(d_fake_data.t()))

            # torch.zeros(1)与torch.zeros([1]) 的 size 均为 torch.Size([1])
            d_fake_error = loss(torch.sum(d_fake_decision).reshape(1, ), torch.zeros([1]).to(device))
            d_fake_error.backward()

            # 仅优化 D 的参数, 基于backward()在网络结构中填充的梯度(grad)更新权重
            d_optimizer.step()

            # 提取 鉴别器D 对于real data的损失值和fake data的损失值;
            # dre 应随着迭代趋近于 dfe
            dre, dfe = extract(d_real_error)[0], extract(d_fake_error)[0]

        # 训练生成器(Generator)
        for g_index in range(g_steps):
            # 在训练 G 的过程中, 保持 D 不动(此时, D已具有一定的鉴别能力)
            # 在 D 有一定鉴别能力的基础上, 训练 G
            G.zero_grad()

            # 按Generator的输入大小，采样mini-batch个输入
            gen_input = g_sampler(mini_batch_size, g_input_size)

            gen_input = gen_input.to(device)

            # G 生成假的数据分布
            g_fake_data = G(gen_input)

            dg_fake_decision = D(preprocess(g_fake_data.t()))

            # loss_fn: G 通过迭代使生成的假的数据分布让 D 打更高分; 即, 使 G 生成的数据更靠近 D 的判别标准
            g_error = loss(torch.sum(dg_fake_decision).reshape(1, ), torch.ones([1]).to(device))
            g_error.backward()

            # 仅优化 生成器G 的参数
            g_optimizer.step()

            # 提取 生成器G 的损失
            ge = extract(g_error)[0]

        if epoch % print_interval == 0:
            print("Epoch %s: D (%s real_err, %s fake_err) G (%s err); Real Dist (%s),  Fake Dist (%s) " %
                  (epoch, dre, dfe, ge, get_mean_and_std(extract(d_real_data)), get_mean_and_std(extract(d_fake_data))))

            d_fake_list.append([extract(d_fake_data), epoch])

        if epoch % plot_interval == 0:
            # 绘制生成器G生成的数据分布图
            plot_generated_distribution(data_type=data_type, generated_fake_data=g_fake_data)

    # 绘制生成器G生成的数据分布(distribution)的均值(mean)和方差(std)
    plot_mean_std(d_fake_list)

    # 将生成器G生成的数据分布写入文件
    write_data('d_fake_list.txt', d_fake_list)


train()
