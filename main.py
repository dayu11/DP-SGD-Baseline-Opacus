import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision

import os
import argparse
import csv
import random
import time
import numpy as np

from models import wrn16_4, wrn40_4
from utils import get_data_loader

#package for computing individual gradients
from opacus.accountants.utils import get_noise_multiplier
from opacus import PrivacyEngine

parser = argparse.ArgumentParser(description='Differentially Private learning with DP-SGD')

## general arguments
parser.add_argument('--dataset', default='cifar10', type=str, help='dataset name')
parser.add_argument('--model', default='wrn40_4', type=str, help='model name [wrn16_4, wrn40_4]')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--sess', default='debug', type=str, help='session name')
parser.add_argument('--seed', default=-1, type=int, help='random seed')
parser.add_argument('--weight_decay', default=0., type=float, help='weight decay')

parser.add_argument('--batchsize', default=1000, type=int, help='batch size')
parser.add_argument('--accmu', default=5, type=int, help='number of gradient accumulations')

parser.add_argument('--n_epoch', default=300, type=int, help='total number of epochs')
parser.add_argument('--lr', default=4, type=float, help='base learning rate (default=0.1)')
parser.add_argument('--momentum', default=0., type=float, help='value of momentum')

## arguments for learning with differential privacy
parser.add_argument('--private', '-p', action='store_true', help='enable differential privacy')
parser.add_argument('--clip', default=1, type=float, help='gradient clipping bound')
parser.add_argument('--eps', default=8., type=float, help='privacy parameter epsilon')
parser.add_argument('--delta', default=1e-5, type=float, help='desired delta')

args = parser.parse_args()

c10_class_mapping = ['Air.', 'Auto.', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

assert args.dataset in ['cifar10'] # no tested for other datasets
assert args.model in ['wrn40_4', 'wrn16_4']

use_cuda = True
best_acc = 0  
start_epoch = 0  
batch_size = args.batchsize

if(args.seed != -1): 
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

print('==> Preparing data..')
trainloader, testloader, n_training, n_test = get_data_loader(args.dataset, batchsize = args.batchsize)
train_samples, train_labels = None, None

print('# of training examples: ', n_training, '# of testing examples: ', n_test)

in_channel = 3
if(args.dataset == 'mnist'):
    in_channel = 1

print('\n==> Creating ResNet20 model instance')
net = eval('%s(in_channel=in_channel)'%args.model)
net.cuda()

num_params = 0
np_list = []
for p in net.parameters():
    num_params += p.numel()
    np_list.append(p.numel())

optimizer = optim.SGD(
        net.parameters(), 
        lr=args.lr, 
        momentum=args.momentum, 
        weight_decay=args.weight_decay)

print('total number of parameters: ', num_params/(10**6), 'M')

loss_func = nn.CrossEntropyLoss()


if(args.private):

    privacy_engine = PrivacyEngine()


    noise_multiplier = get_noise_multiplier(
        target_epsilon=args.eps,
        target_delta=args.delta,
        sample_rate=args.batchsize * args.accmu / 50000,
        epochs=args.n_epoch
    )

    print('noise multiplier: ', noise_multiplier, ' target eps: ', args.eps)

    net, optimizer, trainloader = privacy_engine.make_private(module=net, optimizer=optimizer, data_loader=trainloader, poisson_sampling=False, noise_multiplier=noise_multiplier, max_grad_norm=args.clip)


scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.n_epoch)

def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    t0 = time.time()
    steps = n_training//args.batchsize

    loader = iter(trainloader)

    num_update = 0

    for batch_idx in range(steps):
        inputs, targets = next(loader)
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()

        optimizer.zero_grad()
        outputs = net(inputs)
        loss = loss_func(outputs, targets)
        loss.backward()

        # for pn, p in net.named_parameters():
        #     real_norm = p.grad.norm().item()
        #     per_example_norm = torch.mean(p.grad_sample, dim=0).norm().item()
        #     print(pn, real_norm-per_example_norm<1e-6)
        # exit()
        num_update += 1
        if(num_update % args.accmu != 0):
            optimizer.signal_skip_step()

        optimizer.step()
        step_loss = loss.item()

        train_loss += step_loss
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).float().cpu().sum()
        acc = 100.*float(correct)/float(total)
    t1 = time.time()
    print('Train loss:%.5f'%(train_loss/(batch_idx+1)), 'time: %d s'%(t1-t0), 'train acc:', acc, end=' ')
    return (train_loss/batch_idx, acc)


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    all_correct = []

    all_class_correct = np.array([0.]*10)


    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            outputs = net(inputs)
            loss = loss_func(outputs, targets)
            step_loss = loss.item()

            test_loss += step_loss 
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct_idx = predicted.eq(targets.data).cpu()

            for j in range(10):
                _idx = targets == j
                all_class_correct[j] += correct_idx[_idx].sum().item()

            all_correct += correct_idx.numpy().tolist()
            correct += correct_idx.sum()

        all_class_correct /= 1000
        acc = 100.*float(correct)/float(total)
        print('test loss:%.5f'%(test_loss/(batch_idx+1)), 'test acc:', acc)
        print('cls wise acc :', end = ' ')
        for i in range(10):
            print('%s: %.2f'%(c10_class_mapping[i], all_class_correct[i]*100), end = ' ')
        print('')

    return (test_loss/batch_idx, acc)


print('\n==> Strat training')

for epoch in range(start_epoch, args.n_epoch):
    train_loss, train_acc = train(epoch)
    scheduler.step()
    test_loss, test_acc = test(epoch)
