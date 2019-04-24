import torch
import torch.nn as nn
from torchsummary import summary

# Code taken from
# https://discuss.pytorch.org/t/flatten-layer-of-pytorch-build-by-sequential-container/5983
# https://discuss.pytorch.org/t/converting-2d-to-1d-module/25796/3
class Flatten(torch.nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x): 
        return x.view(x.size(0), -1)

model1 = nn.Sequential(
            nn.Conv2d(in_channels = 1 , out_channels = 3, kernel_size = 7, padding = 0, stride=2), 
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            Flatten()
        )

model2 = nn.Sequential(
            nn.Conv2d(in_channels = 1 , out_channels = 3, kernel_size = 7, padding = 0, stride=2), 
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            Flatten(),
            nn.Linear(75, 26)
        )

model_zoo = [model1, model2]

for i in range(len(model_zoo)):
    model = model_zoo[i].cuda()
    summary(model, (1,28,28))

