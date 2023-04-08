# %%
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR, MultiStepLR
import numpy as np
import torch.nn as nn
from math import *

# %%
torch.cuda.set_device(1)

# %%
torch.set_default_tensor_type('torch.DoubleTensor') # 设置浮点类型为 torch.float64

# %%
# 定义激活函数: swish(x)
def acti(x):
    return x*torch.sigmoid(x)  

# %%
# 定义网络结构
class DeepRitzNet(torch.nn.Module):
    def __init__(self, input_width, layer_width):
        super(DeepRitzNet, self).__init__()
        self.linear_in = torch.nn.Linear(input_width, layer_width)
        self.linear1 = torch.nn.Linear(layer_width, layer_width)
        self.linear2 = torch.nn.Linear(layer_width, layer_width)
        self.linear3 = torch.nn.Linear(layer_width, layer_width)
        self.linear4 = torch.nn.Linear(layer_width, layer_width)
        self.linear5 = torch.nn.Linear(layer_width, layer_width)
        self.linear6 = torch.nn.Linear(layer_width, layer_width)
        self.linear_out = torch.nn.Linear(layer_width, 1)

    def forward(self, x):
        y = self.linear_in(x) # fully connect layer
        y = y + acti(self.linear2(acti(self.linear1(y)))) # residual block 1
        y = y + acti(self.linear4(acti(self.linear3(y)))) # residual block 2
        y = y + acti(self.linear6(acti(self.linear5(y)))) # residual block 3
        output = self.linear_out(y) # fully connect layer
        return output

# %%
dimension = 4

# %%
# exact solution
def u_ex(x):  
    x_temp = torch.cos(pi*x)
    u_x = (x_temp.sum(1)).reshape([x.size()[0], 1]) # x_temp.sum(1) 按行求和
    return u_x

# %%
def f(x):
    x_temp = torch.cos(pi*x)
    f_x = 2*pi**2*(x_temp.sum(1)).reshape([x.size()[0], 1]) # x_temp.sum(1) 按行求和
    return f_x

# %%
def DRM(x):
    u_hat = model(x)
    ux = torch.zeros(x.size()[0], dimension).cuda()
    step_size = 0.0001
    for i in range(dimension):
        dx = torch.zeros(x.size()[0], dimension).cuda()
        dx[:, i] = torch.ones(x.size()[0])
        ux[:, i] = (model(x+step_size*dx) - model(x-step_size*dx))[:,0]/step_size/2
    uxsq = (torch.sum(ux**2, dim = 1)).reshape([x.size()[0], 1]) # dim = 1 按行求和
    f_temp = f(x)
    part_1 = torch.sum(0.5 * uxsq + 0.5*pi**2*u_hat**2 - f_temp*u_hat)/x.size()[0]
    
    Nb = 100
    xa1 = torch.rand(Nb, dimension).cuda() 
    xa1[:, 0] = torch.zeros(Nb)
    xa2 = torch.rand(Nb, dimension).cuda() 
    xa2[:, 0] = torch.ones(Nb)
    xb1 = torch.rand(Nb, dimension).cuda()
    xb1[:, 1] = torch.zeros(Nb)
    xb2 = torch.rand(Nb, dimension).cuda()
    xb2[:, 1] = torch.ones(Nb)
    xc1 = torch.rand(Nb, dimension).cuda()
    xc1[:, 2] = torch.zeros(Nb)
    xc2 = torch.rand(Nb, dimension).cuda()
    xc2[:, 2] = torch.ones(Nb)
    xd1 = torch.rand(Nb, dimension).cuda()
    xd1[:, 3] = torch.zeros(Nb)
    xd2 = torch.rand(Nb, dimension).cuda()
    xd2[:, 3] = torch.ones(Nb)
    sa1 = (torch.sum((model(xa1)-u_ex(xa1))**2)/xa1.size()[0]).cuda()
    sa2 = (torch.sum((model(xa2)-u_ex(xa2))**2)/xa1.size()[0]).cuda() 
    sb1 = (torch.sum((model(xb1)-u_ex(xb1))**2)/xa1.size()[0]).cuda() 
    sb2 = (torch.sum((model(xb2)-u_ex(xb2))**2)/xa1.size()[0]).cuda()
    sc1 = (torch.sum((model(xc1)-u_ex(xc1))**2)/xa1.size()[0]).cuda() 
    sc2 = (torch.sum((model(xc2)-u_ex(xc2))**2)/xa1.size()[0]).cuda()
    sd1 = (torch.sum((model(xd1)-u_ex(xd1))**2)/xa1.size()[0]).cuda() 
    sd2 = (torch.sum((model(xd2)-u_ex(xd2))**2)/xa1.size()[0]).cuda()
    part_2 = sa1 + sa2 + sb1 + sb2 + sc1 + sc2 + sd1 + sd2
    
    lambda1 = 100.0
    return part_1 + lambda1 * part_2 / 8

# %%
Data_size = 2000
def Gendata():
    x = torch.rand(Data_size, dimension)
    return x.cuda()

# %%
# 正态分布初始化参数
def initparam(model, sigma):
    for m in model.modules():
        if isinstance(m, nn.Linear):
            m.weight.data.normal_(0, sigma)
    return model

# %%
model = DeepRitzNet(dimension, 8)
model = initparam(model, 0.5)

# %%
device = torch.device("cuda:1" )
model.to(device)

# %%
import torch.optim as optim
import torch.nn as nn
import time

# %%
def relative_error():
    x = Gendata()
    predict = model(x)
    exact = u_ex(x)
    value = torch.sqrt(torch.sum((predict - exact )**2))/torch.sqrt(torch.sum((exact )**2))
    return value

# %%
traintime = 20000
error_save = np.zeros(traintime)
optimizer = optim.Adam(model.parameters())

# %%
time_start = time.time()
for i in range(traintime):
    optimizer.zero_grad()
    x = Gendata()
    x.requires_grad = True
    losses = DRM(x)
    losses.backward()
    optimizer.step()
    error = relative_error()
    error_save[i] = float(error)
    
    if i % 50 == 0:
        print("current epoch is: ", i)
        print("current loss is: ", losses.detach())
        print("current relative error is: ", error.detach())
        np.save("DRM_relative_error_4D_Dirichlet.npy", error_save)
np.save("DRM_relative_error_4D_Dirichlet.npy", error_save)
time_end = time.time()
print('total time is: ', time_end-time_start, 'seconds')

# %%



