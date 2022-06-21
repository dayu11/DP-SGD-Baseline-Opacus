
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

import os


import numpy as np

from rdp_accountant import compute_rdp, get_privacy_spent


def process_grad_batch(params, clipping=1):
    n = params[0].grad_batch.shape[0]
    grad_norm_list = torch.zeros(n).cuda()
    for p in params: 
        flat_g = p.grad_batch.reshape(n, -1)
        current_norm_list = torch.norm(flat_g, dim=1)
        grad_norm_list += torch.square(current_norm_list)
    grad_norm_list = torch.sqrt(grad_norm_list)
    scaling = clipping/grad_norm_list
    scaling[scaling>1] = 1

    for p in params:
        p_dim = len(p.shape)
        scaling = scaling.view([n] + [1]*p_dim)
        p.grad_batch *= scaling
        p.grad = torch.mean(p.grad_batch, dim=0)
        p.grad_batch.mul_(0.)

def get_data_loader(dataset, batchsize):
    if(dataset == 'svhn'):
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        trainset = torchvision.datasets.SVHN('./data',split='train', download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=73257, shuffle=True, num_workers=0) #load full btach into memory, to concatenate with extra data

        extraset = torchvision.datasets.SVHN('./data',split='extra', download=True, transform=transform)
        extraloader = torch.utils.data.DataLoader(extraset, batch_size=531131, shuffle=True, num_workers=0) #load full btach into memory

        testset = torchvision.datasets.SVHN('./data',split='test', download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batchsize, shuffle=False, num_workers=0)
        return trainloader, extraloader, testloader, len(trainset)+len(extraset), len(testset)
    elif(dataset == 'mnist'):
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        trainset = torchvision.datasets.MNIST('./data', train=True, download=True, transform=transform)#
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batchsize, shuffle=False, num_workers=2)

        testset = torchvision.datasets.MNIST('./data', train=False, download=True, transform=transform)#
        testloader = torch.utils.data.DataLoader(testset, batch_size=batchsize, shuffle=False, num_workers=2)
        return trainloader, testloader, len(trainset), len(testset)
    else:
        transform_train = transforms.Compose([
        # transforms.RandomCrop(32, padding=4),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train) 
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batchsize, shuffle=True, num_workers=2)

        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test) 
        testloader = torch.utils.data.DataLoader(testset, batch_size=2000, shuffle=False, num_workers=2)
        return trainloader, testloader, len(trainset), len(testset)


