# -*- coding: utf-8 -*-
from __future__ import print_function
import os
os.environ['HOROVOD_FUSION_THRESHOLD'] = '0' # default: 64MB
os.environ['HOROVOD_CYCLE_TIME'] = '0' # default: 5ms
os.environ['HOROVOD_NUM_NCCL_STREAMS'] = '1'

import torch
import tcmm
import time
import mpi4py
import random
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

def reduceToallreduce():
    rank = hvd.rank()
    local_rank = hvd.local_rank()
    size = hvd.size()
    torch.cuda.set_device(local_rank)
    communicator = tcmm.Communicator(rank, size, 1)
    
    # params
    L = 20
    M = [1024*256, 1024*512, 1024*1024, 1024*1024*2, 1024*1024*4, 1024*1024*8, 1024*1024*16, 1024*1024*32, 1024*1024*64, 1024*1024*128]
    reduce_times = []
    allreduce_times = []
    # h2d
    tensors = [torch.zeros(m).cuda() for m in M] # note: element_size = 4 Bytes (32 bits)
    
    # allreduce
    for tensor in tensors:
        communicator.allReduce(tensor)
        communicator.synchronize()
        stime = time.time()
        for i in range(L):
            communicator.allReduce(tensor)
            communicator.synchronize()
            #hvd.allreduce(tensor)
        ttime = time.time() - stime
        allreduce_times.append(ttime / L)
    
    # reduce
    for tensor in tensors:
        communicator.reduce(tensor, 0)
        communicator.synchronize()
        stime = time.time()
        for i in range(L):
            communicator.reduce(tensor, 0)
            communicator.synchronize()
        ttime = time.time() - stime
        reduce_times.append(ttime / L)
    
    if hvd.rank() == 0:
        print('allreduce time via tcmm:', allreduce_times)
        print('reduce time via tcmm:', reduce_times)
    

def multiple_allreduce():
    rank = hvd.rank()
    local_rank = hvd.local_rank()
    size = hvd.size()
    torch.cuda.set_device(local_rank)
    communicator = tcmm.Communicator(rank, size, 1)
    
    # params
    L = 128
    M = (1024, 1024) # 4MB
    tensors = [torch.zeros(M).cuda() for i in range(L)]  # h2d
    
    # hvd allreduce
    for tensor in tensors:
        hvd.allreduce(tensor)
    stime = time.time()
    for tensor in tensors:
        hvd.allreduce(tensor)
    ttime = time.time() - stime
    if hvd.rank() == 0:
        print("hvd allreduce:", ttime)
    
    # hvd allreduce_async_ + sync
    for tensor in tensors:
        h = hvd.allreduce_async_(tensor)
        hvd.synchronize(h)
    stime = time.time()
    for tensor in tensors:
        h = hvd.allreduce_async_(tensor)
        hvd.synchronize(h)
    ttime = time.time() - stime
    if hvd.rank() == 0:
        print("hvd allreduce_async_ + sync:", ttime)

    # hvd allreduce_async_
    handles = []
    for tensor in tensors:
        h = hvd.allreduce_async_(tensor)
        handles.append(h)
    for handle in handles:
        hvd.synchronize(handle)
    handles = []
    stime = time.time()
    for tensor in tensors:
        h = hvd.allreduce_async_(tensor)
        handles.append(h)
    for handle in handles:
        hvd.synchronize(handle)
    ttime = time.time() - stime
    if hvd.rank() == 0:
        print("hvd allreduce_async_:", ttime)


def multiple_tcmm():
    rank = hvd.rank()
    local_rank = hvd.local_rank()
    size = hvd.size()
    torch.cuda.set_device(local_rank)
    communicator = tcmm.Communicator(rank, size, 1)
    
    # params
    L = 128
    M = (1024, 1024)
    tensors = [torch.zeros(M).cuda() for i in range(L)]  # h2d

    # init comm
    tensor = torch.rand(2).cuda()
    communicator.allReduce(tensor)
    communicator.synchronize()
    
    # tcmm allreduce + sync
    for tensor in tensors:
        communicator.allReduce(tensor)
        communicator.synchronize()
    stime = time.time()
    for tensor in tensors:
        communicator.allReduce(tensor)
        communicator.synchronize()
    ttime = time.time() - stime
    if hvd.rank() == 0:
        print("tcmm allreduce + sync:", ttime)

    # tcmm allreduce
    for tensor in tensors:
        communicator.allReduce(tensor)
    communicator.synchronize()
    stime = time.time()
    for tensor in tensors:
        communicator.allReduce(tensor)
    communicator.synchronize()
    ttime = time.time() - stime
    if hvd.rank() == 0:
        print("tcmm allreduce:", ttime)

    # tcmm reduce + sync
    for tensor in tensors:
        communicator.reduce(tensor, 0)
        communicator.synchronize()
    stime = time.time()
    for tensor in tensors:
        communicator.reduce(tensor, 0)
        communicator.synchronize()
    ttime = time.time() - stime
    if hvd.rank() == 0:
        print("tcmm reduce + sync:", ttime)

    # tcmm reduce
    for tensor in tensors:
        communicator.reduce(tensor, 0)
    communicator.synchronize()
    stime = time.time()
    for tensor in tensors:
        communicator.reduce(tensor, 0)
    communicator.synchronize()
    ttime = time.time() - stime
    if hvd.rank() == 0:
        print("tcmm reduce:", ttime)

def merged_comm():
    rank = hvd.rank()
    local_rank = hvd.local_rank()
    size = hvd.size()
    torch.cuda.set_device(local_rank)
    communicator = tcmm.Communicator(rank, size, 1)
    
    # params
    L = 128
    M = (1024, 1024)
    tensor = torch.zeros(M).cuda()
    tensor_merged = torch.zeros(M + (L,)).cuda()

    # init comm
    communicator.allReduce(tensor)
    communicator.synchronize()
    hvd.allreduce(tensor)
    
    # hvd allreduce_async_
    stime = time.time()
    handles = []
    for i in range(L):
        handles.append(hvd.allreduce_async_(tensor))
    for handle in handles:
        hvd.synchronize(handle)
    if hvd.rank() == 0:
        print('allreduce time via hvd.allreduce_async_:', time.time() - stime)
    
    # tcmm allReduce
    stime = time.time()
    for i in range(L):
        communicator.allReduce(tensor)
    communicator.synchronize()
    if hvd.rank() == 0:
        print('allreduce time via tcmm:', time.time() - stime)
    
    # tcmm reduce
    stime = time.time()
    for i in range(L):
        communicator.reduce(tensor, 0)
    communicator.synchronize()
    if hvd.rank() == 0:
        print('reduce time via tcmm:', time.time() - stime)
    
    # merged hvd allreduce_async_
    stime = time.time()
    h = hvd.allreduce_async_(tensor_merged)
    hvd.synchronize(h)
    if hvd.rank() == 0:
        print('allreduce time via hvd.allreduce_async_ merged:', time.time() - stime)
    
    # merged tcmm allReduce
    stime = time.time()
    communicator.allReduce(tensor_merged)
    communicator.synchronize()
    if hvd.rank() == 0:
        print('allreduce time via tcmm merged:', time.time() - stime)
    
    # merged tcmm reduce
    stime = time.time()
    communicator.reduce(tensor_merged, 0)
    communicator.synchronize()
    if hvd.rank() == 0:
        print('reduce time via tcmm merged:', time.time() - stime)
    
def reduce_placement():
    rank = hvd.rank()
    local_rank = hvd.local_rank()
    size = hvd.size()
    torch.cuda.set_device(local_rank)
    communicator = tcmm.Communicator(rank, size, 1)
    
    # params
    L = 16
    M = (1024, 1024, 128)
    tensors = [torch.zeros(M).cuda() for i in range(L)]  # h2d

    # init comm
    tensor = torch.rand(2).cuda()
    communicator.allReduce(tensor)
    communicator.synchronize()
    
    # allreduce
    stime = time.time()
    for tensor in tensors:
        communicator.allReduce(tensor)
    communicator.synchronize()
    ttime = time.time() - stime
    if hvd.rank() == 0:
        print("allreduce:", ttime)
    
    # reduce to 0
    stime = time.time()
    for tensor in tensors:
        communicator.reduce(tensor, 0)
    communicator.synchronize()
    ttime = time.time() - stime
    if hvd.rank() == 0:
        print("reduce to 0:", ttime)

    # reduce to n-1
    stime = time.time()
    for tensor in tensors:
        communicator.reduce(tensor, hvd.size() - 1)
    communicator.synchronize()
    ttime = time.time() - stime
    if hvd.rank() == hvd.size() - 1:
        print("reduce to n-1:", ttime)

    # reduce in turn
    stime = time.time()
    for i, tensor in enumerate(tensors):
        communicator.reduce(tensor, i % hvd.size())
    communicator.synchronize()
    ttime = time.time() - stime
    if hvd.rank() == (i % hvd.size()):
        print("reduce in turn:", ttime)

    # reduce in random
    random.seed(42) # set seed to guarantee to consistency
    workers = [random.choice(range(hvd.size())) for i in range(L)]
    stime = time.time()
    for i, tensor in enumerate(tensors):
        communicator.reduce(tensor, workers[i])
    communicator.synchronize()
    ttime = time.time() - stime
    if hvd.rank() == workers[i]:
        print("reduce in random:", ttime)


if __name__ == '__main__':
    #allreduce()
    #multi_bcast()
    #reduce()
    #reduceToallreduce()
    #multiple_allreduce()
    multiple_tcmm()
    #merged_comm()
    #reduce_placement()
