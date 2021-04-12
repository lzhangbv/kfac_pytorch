from __future__ import print_function
import argparse
import time
import os
import sys
import datetime
import math
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s [%(filename)s:%(lineno)d] %(levelname)s %(message)s')

strhdlr = logging.StreamHandler()
strhdlr.setFormatter(formatter)
logger.addHandler(strhdlr) 

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torchvision import datasets, transforms, models

import cifar_resnet as resnet
from utils import *
import kfac
import horovod.torch as hvd

def initialize():
    # Training Parameters
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Example')
    parser.add_argument('--model', type=str, default='resnet32',
                        help='ResNet model to use [20, 32, 56]')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--test-batch-size', type=int, default=128, metavar='N',
                        help='input batch size for testing (default: 128)')
    parser.add_argument('--epochs', type=int, default=200, metavar='N',
                        help='number of epochs to train (default: 200)')

    # SGD Parameters
    parser.add_argument('--base-lr', type=float, default=0.1, metavar='LR',
                        help='base learning rate (default: 0.1)')
    parser.add_argument('--lr-decay', nargs='+', type=int, default=[100, 150],
                        help='epoch intervals to decay lr')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=5e-4, metavar='W',
                        help='SGD weight decay (default: 5e-4)')
    parser.add_argument('--warmup-epochs', type=int, default=5, metavar='WE',
                        help='number of warmup epochs (default: 5)')


    # KFAC Parameters
    parser.add_argument('--kfac-type', type=str, default='F1mc', 
                        help='choices: F1mc or Femp') 
    parser.add_argument('--kfac-name', type=str, default='inverse',
                        help='choices: %s' % kfac.kfac_mappers.keys() + ', default: '+'inverse')
    parser.add_argument('--exclude-parts', type=str, default='',
                        help='choices: ComputeFactor,CommunicateFactor,ComputeInverse,CommunicateInverse')
    parser.add_argument('--kfac-update-freq', type=int, default=10,
                        help='iters between kfac inv ops (0 for no kfac updates) (default: 10)')
    parser.add_argument('--kfac-cov-update-freq', type=int, default=1,
                        help='iters between kfac cov ops (default: 1)')
    parser.add_argument('--stat-decay', type=float, default=0.95,
                        help='Alpha value for covariance accumulation (default: 0.95)')
    parser.add_argument('--damping', type=float, default=0.003,
                        help='KFAC damping factor (defaultL 0.003)')
    parser.add_argument('--kl-clip', type=float, default=0.001,
                        help='KL clip (default: 0.001)')

    # Other Parameters
    parser.add_argument('--log-dir', default='./logs',
                        help='log directory')
    parser.add_argument('--dir', type=str, default='/datasets/cifar10', metavar='D',
                        help='directory to download cifar10 dataset to')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=42, metavar='S',
                        help='random seed (default: 42)')
    parser.add_argument('--fp16-allreduce', action='store_true', default=False,
                        help='use fp16 compression during allreduce')

    args = parser.parse_args()

    # Training Settings
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.use_kfac = True if args.kfac_update_freq > 0 else False
    
    hvd.init()

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.set_device(hvd.local_rank())
        torch.cuda.manual_seed(args.seed)

    torch.backends.cudnn.benchmark = True

    # Logging Settings
    os.makedirs(args.log_dir, exist_ok=True)
    logfile = os.path.join(args.log_dir,
        'cifar10_{}_ep{}_bs{}_kfac{}_{}_gpu{}.log'.format(args.model, args.epochs, args.batch_size, args.kfac_update_freq, args.kfac_name, hvd.size()))

    hdlr = logging.FileHandler(logfile)
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr) 

    args.verbose = True if hvd.rank() == 0 else False
    if args.verbose:
        logger.info(args)
    
    return args


def get_dataset(args):
    # Load Cifar10
    torch.set_num_threads(4)
    kwargs = {'num_workers': 4, 'pin_memory': True} if args.cuda else {}

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    train_dataset = datasets.CIFAR10(root=args.dir, train=True, 
                                     download=False, transform=transform_train)
    test_dataset = datasets.CIFAR10(root=args.dir, train=False,
                                    download=False, transform=transform_test)

    # Use DistributedSampler to partition the training data.
    train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, num_replicas=hvd.size(), rank=hvd.rank())
    train_loader = torch.utils.data.DataLoader(train_dataset,
            batch_size=args.batch_size, sampler=train_sampler, **kwargs)

    # Use DistributedSampler to partition the test data.
    test_sampler = torch.utils.data.distributed.DistributedSampler(
            test_dataset, num_replicas=hvd.size(), rank=hvd.rank())
    test_loader = torch.utils.data.DataLoader(test_dataset, 
            batch_size=args.test_batch_size, sampler=test_sampler, **kwargs)
    
    return train_sampler, train_loader, test_sampler, test_loader


def get_model(args):
    # ResNet
    if args.model.lower() == "resnet20":
        model = resnet.resnet20()
    elif args.model.lower() == "resnet32":
        model = resnet.resnet32()
    elif args.model.lower() == "resnet44":
        model = resnet.resnet44()
    elif args.model.lower() == "resnet56":
        model = resnet.resnet56()
    elif args.model.lower() == "resnet110":
        model = resnet.resnet110()

    if args.cuda:
        model.cuda()

    # Optimizer
    criterion = nn.CrossEntropyLoss()

    args.base_lr = args.base_lr * hvd.size()
    optimizer = optim.SGD(model.parameters(), 
            lr=args.base_lr, 
            momentum=args.momentum,
            weight_decay=args.weight_decay)

    if args.use_kfac:
        KFAC = kfac.get_kfac_module(args.kfac_name)
        preconditioner = KFAC(model, 
                lr=args.base_lr, 
                factor_decay=args.stat_decay, 
                damping=args.damping, 
                kl_clip=args.kl_clip, 
                fac_update_freq=args.kfac_cov_update_freq, 
                kfac_update_freq=args.kfac_update_freq, 
                exclude_parts=args.exclude_parts)
        kfac_param_scheduler = kfac.KFACParamScheduler(
                preconditioner,
                damping_alpha=1,
                damping_schedule=None,
                update_freq_alpha=1,
                update_freq_schedule=None)
    else:
        preconditioner = None

    # Distributed Optimizer
    compression = hvd.Compression.fp16 if args.fp16_allreduce else hvd.Compression.none
    optimizer = hvd.DistributedOptimizer(optimizer, 
                                         named_parameters=model.named_parameters(),
                                         compression=compression,
                                         op=hvd.Average,
                                         backward_passes_per_step=1)

    if hvd.size() > 1:
        hvd.broadcast_optimizer_state(optimizer, root_rank=0)
        hvd.broadcast_parameters(model.state_dict(), root_rank=0)

    # Learning Rate Schedule
    lrs = create_lr_schedule(hvd.size(), args.warmup_epochs, args.lr_decay)
    lr_scheduler = [LambdaLR(optimizer, lrs)]
    if preconditioner is not None:
        lr_scheduler.append(LambdaLR(preconditioner, lrs))
        lr_scheduler.append(kfac_param_scheduler)

    return model, optimizer, preconditioner, lr_scheduler, criterion

def train(epoch, model, optimizer, preconditioner, lr_scheduler, criterion, train_sampler, train_loader, args):
    model.train()
    train_sampler.set_epoch(epoch)
    
    train_loss = Metric('train_loss')
    train_accuracy = Metric('train_accuracy')
    
    for batch_idx, (data, target) in enumerate(train_loader):
        stime = time.time()

        if args.cuda:
            data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()

        if args.use_kfac:
            preconditioner.set_hook_enabled(True)    # forward and save m_a
        
        output = model(data)

        if args.use_kfac and args.kfac_type == 'F1mc':
            pseudo_labels = generate_pseudo_labels(output)
            pseudo_loss = criterion(output, pseudo_labels)
            pseudo_loss.backward(retain_graph=True) # backward and save m_g (F1mc)
            optimizer.synchronize()
            optimizer.zero_grad()                   # zero pseudo gradients

        loss = criterion(output, target)
        with torch.no_grad():
            train_loss.update(criterion(output, target))
            train_accuracy.update(accuracy(output, target))

        if args.use_kfac and args.kfac_type == 'F1mc':
            preconditioner.set_hook_enabled(False)   # backward and no hook (F1mc), but save m_g (Femp)
        
        loss.backward()

        optimizer.synchronize()
        if args.use_kfac:
            preconditioner.step(epoch=epoch)
        with optimizer.skip_synchronize():
            optimizer.step()

    if args.verbose:
        #logger.info('kfac_update_freq:{}, fac_update_freq:{}'.format(preconditioner.kfac_update_freq, preconditioner.fac_update_freq))
        logger.info("[%d] epoch train loss: %.4f, acc: %.3f" % (epoch, train_loss.avg.item(), 100*train_accuracy.avg.item()))

    for scheduler in lr_scheduler:
        scheduler.step()


def test(epoch, model, criterion, test_loader, args):
    model.eval()
    test_loss = Metric('val_loss')
    test_accuracy = Metric('val_accuracy')
    
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            output = model(data)
            test_loss.update(criterion(output, target))
            test_accuracy.update(accuracy(output, target))
            
    if args.verbose:
        logger.info("[%d][0] evaluation loss: %.4f, acc: %.3f" % (epoch, test_loss.avg.item(), 100*test_accuracy.avg.item()))


if __name__ == '__main__':
    #torch.multiprocessing.set_start_method('spawn')

    args = initialize()

    train_sampler, train_loader, _, test_loader = get_dataset(args)
    model, optimizer, preconditioner, lr_scheduler, criterion = get_model(args)

    start = time.time()

    for epoch in range(args.epochs):
        train(epoch, model, optimizer, preconditioner, lr_scheduler, criterion, train_sampler, train_loader, args)
        test(epoch, model, criterion, test_loader, args)

    if args.verbose:
        logger.info("Training time: %s", str(datetime.timedelta(seconds=time.time() - start)))

