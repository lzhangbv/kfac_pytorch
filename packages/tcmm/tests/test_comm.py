# -*- coding: utf-8 -*-
from __future__ import print_function
import torch
import tcmm
import time
import mpi4py
import horovod.torch as hvd
torch.random.manual_seed(10)
hvd.init()


def allreduce():
    rank = hvd.rank()
    local_rank = hvd.local_rank()
    size = hvd.size()
    torch.cuda.set_device(local_rank)
    communicator = tcmm.Communicator(rank, size)
    tensor = torch.rand(2).cuda()
    print('before rank: %d' % rank, tensor)
    communicator.allReduce(tensor)
    print('after rank: %d' % rank, tensor)

def multi_bcast():
    rank = hvd.rank()
    local_rank = hvd.local_rank()
    size = hvd.size()
    torch.cuda.set_device(local_rank)
    communicator = tcmm.Communicator(rank, size)
    ntensors = 2
    tensors = []
    for i in range(ntensors):
        t = torch.rand(2).cuda()
        tensors.append(t)
    def _op(tensor):
        tensor.mul_(2)
        return None
    print('before rank: %d' % rank, tensors)
    communicator.multiBcast(tensors, _op)
    print('after rank: %d' % rank, tensors)

def reduce():
    rank = hvd.rank()
    local_rank = hvd.local_rank()
    size = hvd.size()
    torch.cuda.set_device(local_rank)
    communicator = tcmm.Communicator(rank, size)
    tensor = torch.rand(2).cuda()
    print('before rank: %d' % rank, tensor)
    communicator.reduce(tensor, 0)
    print('after rank: %d' % rank, tensor)

if __name__ == '__main__':
    #allreduce()
    #multi_bcast()
    reduce()
