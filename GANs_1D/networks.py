
import torch.nn as nn


# 生成器
class Generator(nn.Module):
    """
    所有神经网络(neural network)都需要继承父类(nn.Module), 并实现方法 1. "__init__"; 2. "forward"
    1. __init__：当一个类实例化时, 会自动调用方法 __init__(); 类实例化时的参数会自动传递给 __init__
    2. forward(): 继承nn.Module的类实例化后, 类的实例化变量会直接自动调用forward,
                  此时, 传入的参数会直接传给forward, forward的返回值也会直接返回给对应变量, 即
                  loss = model(input_data) <==> loss = model.forward(input_data)
                  实质：不过是 model 又封装了一层, 可以少写一步.forward(input_data)
    """

    # 类方法的第一个参数必须是self, 可参见 https://docs.python.org/zh-cn/3/tutorial/classes.html
    def __init__(self, input_size, hidden_size, output_size, f):
        # 同Java继承机制(子类继承父类的所有属性和方法, 用父类的初始化方法对继承自父类的属性进行初始化)
        # 首先找到父类, 然后把self转化为父类对象, 最后"被转换"的父类对象调用自己的init函数
        super(Generator, self).__init__()
        # nn.Linear(in_features, out_features, bias)：对输入数据应用线性变换
        # in_feature, 输入样本size; out_features, 输出样本size; bias: 是否添加添加偏置单元
        # 注：map：映射
        self.map1 = nn.Linear(input_size, hidden_size)
        self.map2 = nn.Linear(hidden_size, hidden_size)
        self.map3 = nn.Linear(hidden_size, output_size)
        self.f = f

    def forward(self, x):
        x = self.map1(x)
        x = self.f(x)
        x = self.map2(x)
        x = self.f(x)
        x = self.map3(x)
        return x


# 鉴别器
class Discriminator(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, f):
        super(Discriminator, self).__init__()
        self.map1 = nn.Linear(input_size, hidden_size)
        self.map2 = nn.Linear(hidden_size, hidden_size)
        self.map3 = nn.Linear(hidden_size, output_size)
        self.f = f

    def forward(self, x):
        x = self.map1(x)
        x = self.f(x)
        x = self.map2(x)
        x = self.f(x)
        x = self.map3(x)
        x = self.f(x)
        return x