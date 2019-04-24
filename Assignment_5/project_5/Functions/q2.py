
import argparse
import os
import shutil
import time
import numpy as np
import sys
sys.path.insert(0, './src')

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim


model = nn.Sequential(nn.Linear(2,3), nn.ReLU(), nn.Linear(3,3), nn.ReLU(), nn.Linear(3,2))
print(model)

for m in model.modules():
    if isinstance(m, nn.Linear):
        m.weight.data = torch.ones(m.weight.data.shape)
        m.bias.data.fill_(1.0)

loss = nn.MSELoss(reduction='sum')
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

input  = torch.Tensor([[1, 1]])
target = torch.Tensor([[1, 1]])

model.train()
output  = model(input)
my_loss = loss(output, target)

optimizer.zero_grad()
my_loss.backward()


#print(input)
#print(output)
#print(input.shape)
#print(output.shape)
print("Loss = {}".format(my_loss))

print("\nGradients for the 3rd linear layer")
print(model[4].weight.grad)
print(model[4].bias.grad)

print("\nGradients for the 2nd linear layer")
print(model[2].weight.grad)
print(model[2].bias.grad)

print("\nGradients for the 1st linear layer")
print(model[0].weight.grad)
print(model[0].bias.grad)
