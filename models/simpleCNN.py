# 定义卷积神经网络
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):  # 我们定义网络时一般是继承的torch.nn.Module创建新的子类
    def __init__(self):
        super(CNN, self).__init__()  # 第二、三行都是python类继承的基本操作,此写法应该是python2.7的继承格式,但python3里写这个好像也可以
        self.conv1 = nn.Conv2d(145, 64, (3, 3), stride=(1, 1))  # in_channels, out_channels, kernel_size=5*5
        self.conv2 = nn.Conv2d(64, 128, (3, 3), stride=(1, 1))
        self.conv3 = nn.Conv2d(128, 128, (3, 3), stride=(1, 1))
        self.pool = nn.MaxPool2d((2, 2), stride=(2, 2))  # equal to nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(346122, 256)  # 接着三个全连接层
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 2)
        self.BN2d = nn.BatchNorm2d(num_features=3)

    def forward(self, x):  # 这里定义前向传播的方法，为什么没有定义反向传播的方法呢？这其实就涉及到torch.autograd模块了，
        # 但说实话这部分网络定义的部分还没有用到autograd的知识，所以后面遇到了再讲
        x = self.pool(F.relu(self.conv1(x)))  # F是torch.nn.functional的别名，这里调用了relu函数 F.relu()
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        print(x.shape)
        #x = x.view(x.size(0), -1)
        x = x.view(-1, 346122)  # .view( )是一个tensor的方法，使得tensor改变size但是元素的总数是不变的。
        #  第一个参数-1是说这个参数由另一个参数确定， 比如矩阵在元素总数一定的情况下，确定列数就能确定行数。
        #  那么为什么这里只关心列数不关心行数呢，因为马上就要进入全连接层了，而全连接层说白了就是矩阵乘法，
        #  你会发现第一个全连接层的首参数是16*5*5，所以要保证能够相乘，在矩阵乘法之前就要把x调到正确的size
        # 更多的Tensor方法参考Tensor: http://pytorch.org/docs/0.3.0/tensors.html
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

#  根据网络层的不同定义不同的初始化方式
def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)
    # 也可以判断是否为conv2d，使用相应的初始化方式
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
     # 是否为批归一化层
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

class _3DCNN(nn.Module):  # 我们定义网络时一般是继承的torch.nn.Module创建新的子类
    def __init__(self):
        super(_3DCNN, self).__init__()  # 第二、三行都是python类继承的基本操作,此写法应该是python2.7的继承格式,但python3里写这个好像也可以
        self.conv1 = nn.Conv3d(1, 8, kernel_size=3, stride=1, padding=1)  # in_channels, out_channels, kernel_size=3*3*3
        self.conv2 = nn.Conv3d(8, 16, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv3d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=1)
        self.BN3d1 = nn.BatchNorm3d(num_features=8)  # num_feature为输入数据通道数
        self.BN3d2 = nn.BatchNorm3d(num_features=16)
        self.BN3d3 = nn.BatchNorm3d(num_features=32)
        self.BN3d4 = nn.BatchNorm3d(num_features=64)
        self.BN3d5 = nn.BatchNorm3d(num_features=128)
        self.pool1 = nn.AdaptiveMaxPool3d((61, 73, 61))  # (61,73,61) is output size
        self.pool2 = nn.AdaptiveMaxPool3d((31, 37, 31))
        self.pool3 = nn.AdaptiveMaxPool3d((16, 19, 16))
        self.pool4 = nn.AdaptiveMaxPool3d((8, 10, 8))
        self.pool5 = nn.AdaptiveMaxPool3d((4, 5, 4))
        self.dropout = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(10240, 1300)  # 接着三个全连接层
        self.fc2 = nn.Linear(1300, 50)
        self.fc3 = nn.Linear(50, 2)

    def forward(self, x):
        x = F.relu(self.BN3d1(self.conv1(x)))
        x = self.pool1(x)
        x = F.relu(self.BN3d2(self.conv2(x)))
        x = self.pool2(x)
        x = F.relu(self.BN3d3(self.conv3(x)))
        x = self.pool3(x)
        x = F.relu(self.BN3d4(self.conv4(x)))
        x = self.pool4(x)
        x = F.relu(self.BN3d5(self.conv5(x)))
        x = self.pool5(x)
        x = x.view(x.size(0), -1)   # x.size(0)此处为batch size
        #x = x.view(-1, 173056)  #
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

#  根据网络层的不同定义不同的初始化方式
def weight_init_3d(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)
    # 也可以判断是否为conv3d，使用相应的初始化方式
    elif isinstance(m, nn.Conv3d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
     # 是否为批归一化层
    elif isinstance(m, nn.BatchNorm3d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)