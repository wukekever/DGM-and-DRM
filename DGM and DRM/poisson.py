# # /*
# #  * @Author: Keke Wu
# #  * @Date: 2021-10-10 12:38:35 
# #  * @Last Modified by:   Keke Wu
# #  * @Last Modified time: 2021-10-10 12:38:35 
# #  */

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

import time
import os
import numpy as np

# neural network
class Net(nn.Module):
    def __init__(self, input_size, layers, output_size):
        super(Net, self).__init__()
        self.input_size = input_size
        self.layers = layers
        self.output_size = output_size
        self.layer_in = nn.Linear(input_size, layers[0])
        self.hidden_layers = nn.ModuleList([nn.Linear(layers[i], layers[i+1]) for i in range(len(layers)-1)])
        self.output_layer = nn.Linear(layers[-1], output_size)
          
    def forward(self, tx):
        output = self.layer_in(tx)
        for i, h_i in enumerate(self.hidden_layers):
            output = self.activation(h_i(output))
        output = self.output_layer(output)
        return output
    
    def activation(self, o):
        return torch.tanh(o)
    
    def Xavier_initi(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

dimension = 1
INPUT_SIZE = dimension
LAYERS = [128, 256, 256, 512]
OUTPUT_SIZE = 1

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device_ids = [0]

net = Net(INPUT_SIZE, LAYERS, OUTPUT_SIZE)
net.Xavier_initi()
if torch.cuda.device_count() > 1:    
    net = nn.DataParallel(net, device_ids = device_ids)
    
net.to(device)

# number of paramerters
param_num = sum(neural.numel() for neural in net.parameters())
print("Total number of paramerters in networks is: {}  ".format(param_num))

# save or load param
def save_param(net, path):
    torch.save(net.state_dict(), path)
    
def load_param(net, path):
    if os.path.exists(path):
        net.load_state_dict(torch.load(path)) 
    else:
        print('File does not exist.')

def mkdir(file_dir):
    isExists = os.path.exists(file_dir)
    if not isExists: 
        os.makedirs(file_dir)
        print("file make direction sucessfully!")
    else: 
        print("file has already exists!")

mkdir(file_dir = "./pkl")
mkdir(file_dir = "./record")

# generate data
def generate_sample(data_size, dimension):
    data_train = torch.tensor([])
    while data_train.size()[0] < data_size:
        temp = torch.rand(data_size, dimension) 
        data_train = torch.cat((data_train, temp), 0)
    return data_train

DATA_SIZE = 10000
train_x = generate_sample(DATA_SIZE, dimension)
train_set = TensorDataset(train_x.to(device))
test_set = TensorDataset(train_x.to(device))

# load datasize = 1000 each times over different batches by 0 process workers with shuffle
trainloader = torch.utils.data.DataLoader(train_set, 
                                          batch_size = 1000, 
                                          shuffle = True, 
                                          num_workers = 0, 
                                          pin_memory = False) 

# load datasize = 1000 each times over different batches by 0 process workers with no shuffle
testloader = torch.utils.data.DataLoader(test_set, 
                                         batch_size = 1000, 
                                         shuffle = False, 
                                         num_workers = 0, 
                                         pin_memory = False)

def model(x):
    D_x_0 = torch.prod(x, axis = -1, keepdim = True) 
    D_x_1 = torch.prod(1. - x, axis = -1,  keepdim = True)
    model_u = D_x_0 * D_x_1 * net(x)
    return model_u

# defination of exact solution
def u_ex(x):  
    return torch.sum(torch.sin(np.pi*x), -1, keepdim = True)

# defination of source term
def f_ex(x):
    return np.pi**2 * torch.sum(torch.sin(np.pi*x), -1, keepdim = True)

class Poisson(nn.Module):
    def __init__(self, dimension, model, Iter, name = 'poisson', **kwargs):
        super(Poisson, self).__init__()
        self.d = dimension
        self.model = model
        self.Iter = Iter

    # inputs: x
    def residual(self, inputs):
        # create variables
        var_name = []
        var_value = list(torch.split(inputs, [1 for _ in range(self.d)], dim = -1))
        createVar = locals()
        for i in range(self.d): 
            createVar['x_%s' %i] = var_value[i]
            var_name.append('x_'+ str(i))
        for i in range(self.d): 
            var_name[i] =  var_value[i]
            var_name[i].requires_grad = True
            
        inputs = torch.cat(var_name, -1)
        u = self.solution(inputs)
        f = f_ex(inputs)
        # loss to DRM by auto differential
        ritz = 0.0
        for i in range(self.d):
            du_dx = torch.autograd.grad(outputs = u, 
                                        inputs = var_name[i], 
                                        grad_outputs = torch.ones(u.shape).to(device), 
                                        create_graph = True)[0]
            ritz += du_dx**2
        
        res = torch.mean(0.5*ritz - f * u) 

        # # loss to DGM by auto differential
        # galerkin = 0.0
        # for i in range(self.d):
        #     du_dx = torch.autograd.grad(outputs = u, 
        #                                 inputs = var_name[i], 
        #                                 grad_outputs = torch.ones(u.shape).to(device), 
        #                                 create_graph = True)[0]
        #     d2u_dx2 = torch.autograd.grad(outputs = du_dx, 
        #                                   inputs = var_name[i],
        #                                   grad_outputs = torch.ones(du_dx.shape).to(device),
        #                                   create_graph = True)[0]
        #     galerkin += d2u_dx2
        
        # res = torch.mean((- galerkin - f)**2)

        return res

    def boundary(self, inputs):
        pass
    
        
    def solution(self, inputs):
        sol = self.model(inputs)
        return sol
    
    def train(self, trainloader):
        time_start = time.time()
        print('Begin training.')
        print('')
        loss_record, error_record = torch.as_tensor([]), torch.as_tensor([])
        for it in range(self.Iter):
            for i, data in enumerate(trainloader, 0): 
                batch_x = data[0] # batch_x.requires_grad = True
                optimizer.zero_grad(set_to_none = True)
                loss = self.residual(batch_x)
                error = self.test(testloader)
                loss.backward()
                optimizer.step()
                scheduler.step()
                torch.cuda.empty_cache() # clear memory
                if i % 1 == 0:
                    print('[Iter: {:5d}/{:5d}, Batch: {:}]'.format(it + 1, self.Iter, i + 1))
                    print('[Learning rate: {:.2e}]'.format(optimizer.param_groups[0]['lr']))  
                    print('[Train loss: {:.2e}, Test error: {:.2e}]'.format(loss.item(), error.item())) 
                loss_record = torch.cat((loss_record, torch.as_tensor([loss])),0)
                error_record = torch.cat((error_record, torch.as_tensor([error])),0)
        
        save_param(net, path = './pkl/net_params.pkl')
        torch.save(loss_record, './record/loss_record.pt')
        torch.save(error_record, './record/error_record.pt')
        # Loss = torch.load('./record/loss_record.pt')
        # Error = torch.load('./record/error_record.pt')

        print('')       
        print('Finished training.')
        time_end = time.time()
        print('Total time is: {:.2e}'.format(time_end - time_start), 'seconds')
        
    def test(self, testloader):
        with torch.no_grad():
            err_up, err_down = 0, 0
            for i, data in enumerate(testloader, 0): # return order i from 0 term to len(testloader) in trainloader
                batch_x = data[0] # batch size
                batch_x.requires_grad = False
                approx = self.model(batch_x)
                ex = u_ex(batch_x)
                err_up_i = torch.sum((approx - ex)**2)
                err_down_i = torch.sum(ex**2)
                err_up += err_up_i
                err_down += err_down_i
            err = (err_up / err_down)**0.5
        return err

# set optimizer and learning rate decay
optimizer = optim.Adam(net.parameters(), lr = 1e-3)
scheduler = lr_scheduler.StepLR(optimizer, 200, 0.96) # every 200 epoch, learning rate * 0.96

Iter = 100
equation = Poisson(dimension = dimension, model = model, Iter = Iter)
equation.train(trainloader)

