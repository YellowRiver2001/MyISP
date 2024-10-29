import torch
import torchvision.datasets as datasets 
import torchvision.transforms as transforms
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
import os

# 设置torch为确定性，确保每次运行的结果相同
_ = torch.manual_seed(0)

# 定义数据预处理流程，包括将数据转换为张量并标准化
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

# 加载MNIST训练集
mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
# 创建训练数据加载器
train_loader = torch.utils.data.DataLoader(mnist_trainset, batch_size=10, shuffle=True)

# 加载MNIST测试集
mnist_testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
# 创建测试数据加载器
test_loader = torch.utils.data.DataLoader(mnist_testset, batch_size=10, shuffle=True)

# 定义设备为CPU
device = "cpu"

# 定义一个非常简单的神经网络
class VerySimpleNet(nn.Module):
    def __init__(self, hidden_size_1=100, hidden_size_2=100):
        super(VerySimpleNet, self).__init__()
        # 定义量化Stub，用于量化输入
        self.quant = torch.quantization.QuantStub()
        # 定义第一个全连接层
        self.linear1 = nn.Linear(28*28, hidden_size_1)
        # 定义第二个全连接层
        self.linear2 = nn.Linear(hidden_size_1, hidden_size_2)
        # 定义第三个全连接层
        self.linear3 = nn.Linear(hidden_size_2, 10)
        # 定义ReLU激活函数
        self.relu = nn.ReLU()
        # 定义反量化Stub，用于反量化输出
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, img):
        # 将图像展平成一维向量
        x = img.view(-1, 28*28)
        # 量化输入数据
        x = self.quant(x)
        # 通过第一个全连接层并激活
        x = self.relu(self.linear1(x))
        # 通过第二个全连接层并激活
        x = self.relu(self.linear2(x))
        # 通过第三个全连接层
        x = self.linear3(x)
        # 反量化输出数据
        x = self.dequant(x)
        return x

# 实例化网络并移动到指定设备
net = VerySimpleNet().to(device)

# 设置量化配置
net.qconfig = torch.ao.quantization.default_qconfig
# model.qconfig = torch.quantization.get_default_qat_qconfig('x86')

net.train()
# 准备量化感知训练（QAT），插入观察者
net_quantized = torch.ao.quantization.prepare_qat(net)
net_quantized

# 定义训练函数
def train(train_loader, net, epochs=5, total_iterations_limit=None):
    # 定义交叉熵损失函数
    cross_el = nn.CrossEntropyLoss()
    # 定义Adam优化器
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

    total_iterations = 0

    for epoch in range(epochs):
        net.train()

        loss_sum = 0
        num_iterations = 0

        # 使用 tqdm 显示训练进度
        data_iterator = tqdm(train_loader, desc=f'Epoch {epoch+1}')
        if total_iterations_limit is not None:
            data_iterator.total = total_iterations_limit
        for data in data_iterator:
            num_iterations += 1
            total_iterations += 1
            x, y = data
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            # 前向传播
            output = net(x.view(-1, 28*28))
            # 计算损失
            loss = cross_el(output, y)
            loss_sum += loss.item()
            avg_loss = loss_sum / num_iterations
            data_iterator.set_postfix(loss=avg_loss)
            # 反向传播
            loss.backward()
            optimizer.step()

            # 如果达到迭代次数限制则停止训练
            if total_iterations_limit is not None and total_iterations >= total_iterations_limit:
                return

# 定义打印模型大小的函数
def print_size_of_model(model):
    torch.save(model.state_dict(), "temp_delme.p")
    print('Size (KB):', os.path.getsize("temp_delme.p")/1e3)
    os.remove('temp_delme.p')

# 训练模型（只训练一个epoch）
train(train_loader, net_quantized, epochs=1)


def test(model: nn.Module, total_iterations: int = None):
    correct = 0
    total = 0

    iterations = 0

    model.eval()

    with torch.no_grad():
        for data in tqdm(test_loader, desc='Testing'):
            x, y = data
            x = x.to(device)
            y = y.to(device)
            # print(x.type,x)
            # 前向传播
            output = model(x.view(-1, 784))
            # 计算预测结果
            for idx, i in enumerate(output):
                if torch.argmax(i) == y[idx]:
                    correct +=1
                total +=1
            iterations += 1
            if total_iterations is not None and iterations >= total_iterations:
                break
    print(f'Accuracy: {round(correct/total, 3)}')

# 检查各层的统计信息
print(f'Check statistics of the various layers')
print(net_quantized)

# # 打印量化前模型的权重矩阵
# print('Weights before quantization')
# print(torch.int_repr(net_quantized.linear1.weight()))

# 将模型转换为量化模型
net_quantized.eval()
net_quantized = torch.ao.quantization.convert(net_quantized)

# 检查各层的统计信息
print(f'Check statistics of the various layers')
print(net_quantized)

# 打印量化后模型的权重矩阵
print('Weights after quantization')
print(torch.int_repr(net_quantized.linear1.weight()))

# 打印量化后模型的大小
print('Size of the model after quantization')
print_size_of_model(net_quantized)

# 测试量化后的模型
print('Testing the model after quantization')
test(net_quantized)
