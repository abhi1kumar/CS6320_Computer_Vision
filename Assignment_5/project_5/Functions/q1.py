"""
    Calculates Validation accuracies on two different fully connected 
    MNIST architectures.
	Version 1 2018/04/01 Abhinav Kumar u1209853

    Code borrowed from
    https://github.com/pytorch/examples/blob/master/mnist/main.py
"""


import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn.functional as F
import numpy as np


################################################################################
# main function
################################################################################
def main():
    #global args, best_prec1
    #args = parser.parse_args()

    fcnn            = True
    cudnn.benchmark = True
    use_cuda        = True
    device          = torch.device("cuda" if use_cuda else "cpu")
    output_folder   = os.path.join(os.getcwd(), 'Outputs')
    input_folder    = os.path.join(os.getcwd(), 'Inputs')
    save_frequency  = 30
    workers         = 4

    # Optimisation Parameters
    lr       = 0.01
    momentum = 0.9
    wd       = 0.003
    epochs   = 121
    lr_decay_step_size = 50
    lr_decay_gamma = 0.1
    batch_size      = 64
    test_batch_size = 10000

    # Preparing splits - train and val splits
    normalize = transforms.Normalize(mean=[0], std=[1])
    transform_list=[transforms.ToTensor(), normalize]            

    val_loader = torch.utils.data.DataLoader(
        datasets.MNIST(root = input_folder, train = False, transform=transforms.Compose(transform_list), download = True),
        batch_size = test_batch_size, shuffle = False, num_workers = workers, pin_memory = True)

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(root= input_folder, train = True, transform=transforms.Compose(transform_list), download = True),
        batch_size = batch_size, shuffle = True,       num_workers = workers, pin_memory = True)

    model_1   = nn.Sequential(nn.Linear(784,10), nn.ReLU(), nn.Linear(10,10), nn.ReLU(), nn.Linear(10,10), nn.ReLU(), nn.Linear(10,10), nn.ReLU(), nn.Linear(10,10))
    model_2   = nn.Sequential(nn.Linear(784,10), nn.ReLU(), nn.Linear(10,10), nn.ReLU(), nn.Linear(10,10), nn.ReLU(), nn.Linear(10,10))
    model_zoo = [model_1, model_2]

    for i in range(len(model_zoo)):
        print("\nModel %d ..." %(i+1))
        model = model_zoo[i]
        print(model)

        # Save directory
        save_dir = os.path.join(output_folder, "q1_model" + str(i+1))

        # Loss function
        criterion = nn.CrossEntropyLoss()

        # Optimizer
        optimizer = torch.optim.SGD(model.parameters(), lr, momentum = momentum,  weight_decay= wd)

        # Learning Rate Scheduler
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = lr_decay_step_size, gamma = lr_decay_gamma)


        print("\nModel save folder  = %s"   %(save_dir))
        print("-------------------------------")
        print("Optimisation Parameters")
        print("-------------------------------")
        print("lr                 = %.4f"   %(lr))
        print("momentum           = %.4f"   %(momentum))
        print("weight decay       = %.4f"   %(wd))
        print("epochs             = %d"     %(epochs))
        print("lr decay step size = %d"     %(lr_decay_step_size))
        print("lr decay gamma     = %.2f\n" %(lr_decay_gamma))

        # Initialize model weights 
        for m in model.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data)

        model     = model.to(device)
        criterion = criterion.to(device)

        # Start training
        for epoch in range(epochs):
            scheduler.step()     
           
            # train for one epoch
            train(train_loader, model, criterion, optimizer, epoch,  device, fcnn)

            # evaluate on validation set
            if (epoch % save_frequency == 0):
                prec1 = validate(val_loader, model, criterion, epoch, device, fcnn)

                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'best_prec1': prec1,
                }, True, filename=os.path.join(save_dir, 'checkpoint_{}.tar'.format(epoch)))

                




################################################################################
# Run one train epoch
################################################################################
def train(train_loader, model, criterion, optimizer, epoch,  device, fcnn):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()

    for i, (data, target) in enumerate(train_loader):
        
        # measure data loading time
        data_time.update(time.time() - end)
        
        # Send the data and label to the device
        data, target = data.to(device), target.to(device)
        
        if fcnn:
            data = data.view(data.shape[0],-1)


        # compute output
        output = model(data)

        # compute gradient and do SGD step
        optimizer.zero_grad()

        loss   = criterion(output, target)

        # Calculate gradients of model in backward pass
        loss.backward()
        optimizer.step()

        output = output.float()
        loss = loss.float()

        # measure accuracy and record loss
        prec1 = accuracy(output.data , target)[0]
        losses.update   (loss.item() , data.size(0))
        top1.update     (prec1.item(), data.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()


################################################################################
# Run evaluation or validation
################################################################################
def validate(val_loader, model, criterion, epoch, device, fcnn):

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (data, target) in enumerate(val_loader):
        
        # Send the data and label to the device
        data, target = data.to(device), target.to(device)

        if fcnn:           
            data = data.view(data.shape[0],-1)
            
        # compute output
        output = model(data)
        loss = criterion(output, target)

        output = output.float()
        loss = loss.float()

        # measure accuracy and record loss
        prec1 = accuracy(output.data , target)[0]
        losses.update   (loss.item() , data.size(0))
        top1.update     (prec1.item(), data.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()


        print('Test: [{0}/{1}]\t'
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
              'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                  i, len(val_loader), batch_time=batch_time, loss=losses,
                  top1=top1))

    print('Epoch: [{0}] * Prec@1 {top1.avg:.3f}'
          .format(epoch, top1=top1))

    return top1.avg


################################################################################
# Save the training model
################################################################################
def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)


################################################################################
# Computes and stores the average and current value
################################################################################
class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


################################################################################
# Computes the precision@k for the specified values of k
################################################################################
def accuracy(output, target, topk=(1,)):

    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

if __name__ == '__main__':
    main() 
