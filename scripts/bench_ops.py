# -*- coding: utf-8 -*-
from __future__ import print_function
import torch
import time
import numpy as np

import tcmm
import reader
import torchsso

def add_value_to_diagonal(X, value):
    return X.add_(torch.diag(X.new(X.shape[0]).fill_(value)))

def compute_eigen(matrix, output):
    A = matrix
    #d, Q = torch.qr(A)
    #d = torch.cholesky(A); Q=None
    add_value_to_diagonal(A, 0.002)
    #d = torch.inverse(A); Q=None
    #d = torchsso.utils.inv(A); Q=None
    #if output is not None:
    #    output.copy_(d)
    d, Q = tcmm.f_symeig(A)
    Q = Q.transpose(-2, -1)
    #d, Q = torch.symeig(A, eigenvectors=True)
    #eps = 1e-10  # for numerical stability
    #d = torch.mul(d, (d > eps).float())
    return None

def bench_ops(n, num_iters, warmup=5):
    a = torch.rand(n).float().cuda()
    a = a.view(-1, a.size(-1))
    #print('a shape: ', a.shape)
    A = a.t() @ (a)
    #A = torch.randn(n, n).float().cuda()
    #A = torch.mm(A, A.t())
    #print('A shape: ', A.shape)
    for i in range(warmup):
        compute_eigen(A, A)
    torch.cuda.synchronize()
    stime = time.time()
    for i in range(num_iters):
        compute_eigen(A, A)
    torch.cuda.synchronize()
    etime = time.time()
    time_used = (etime-stime)/num_iters
    return time_used


def bench_gemm(m, n, num_iters, warmup=5):
    TENSOR_CORE=True
    a = torch.rand(m, n).float().cuda()
    #a = a.view(-1, a.size(-1))
    #print('a shape: ', a.shape)
    for i in range(warmup):
        if TENSOR_CORE:
            tcmm.f_gemm_ex(a.t(), a)
        else:
            A = a.t() @ (a)
    torch.cuda.synchronize()
    stime = time.time()
    for i in range(num_iters):
        if TENSOR_CORE:
            tcmm.f_gemm_ex(a.t(), a)
        else:
            A = a.t() @ (a)
    torch.cuda.synchronize()
    etime = time.time()
    time_used = (etime-stime)/num_iters
    return time_used


def bench():
    ns = range(2048, 8192, 1024) 
    #ns = range(64, 512+64, 64) 
    #ns = range(6272, 8192*2, 1024) 
    #ns = range(3, 512+64, 64) 
    #ns = [3]
    #ns = ns+range(2**20, 2**29, 2**20) 
    #ns = range(2**20, 2**29, 2**20) 
    for n in ns:
        num_iters = 50
        if n > 2**19:
            num_iters = 10
        t = bench_ops(n, num_iters)
        #t = bench_gemm(n, n, num_iters)
        print('%d,%f'%(n,t))

def bench_from_log():
    #logfile = './logs/resnet50-matrixsize-A.log';bs=1;target_bs=1
    #logfile = './logs/resnet34-matrixsize.log';bs=1;target_bs=1
    logfile = './logs/resnet50-matrixsize-ag.log';bs=8;target_bs=32
    workloads = reader.read_tensor_sizes(logfile)
    total_time = []
    num_iters = 50
    total_sizes = []
    for w in workloads:
        m = w[0]*target_bs//bs
        n = w[1]
        t = bench_gemm(m, n, num_iters)
        #t = bench_ops(n, num_iters)
        total_time.append(t)
        total_sizes.append(m*n)
        print('(%d,%d),%f'%(m,n,t))
    print('Log file: ', logfile)
    print('# of Tensors: ', len(total_sizes))
    print('Total size: ', np.sum(total_sizes))
    print('Total time: ', np.sum(total_time))
    print('Max-min-mean-std: ', np.max(total_time), np.min(total_time), np.mean(total_time), np.std(total_time))
    
def bench_customize_comm():
    import horovod.torch as hvd
    torch.random.manual_seed(10)
    hvd.init()
    rank = hvd.rank()
    local_rank = hvd.local_rank()
    size = hvd.size()
    torch.cuda.set_device(local_rank)

    logfile = './logs/resnet50-matrixsize-A.log'
    workloads = reader.read_tensor_sizes(logfile)
    tensors = []
    outputs = []
    for w in workloads:
        n = w[0]
        a = torch.rand(n).float().cuda()
        a = a.view(-1, a.size(-1))
        A = a.t() @ (a)
        tensors.append(A)
        outputs.append(A.new_zeros(A.shape))

        communicator = tcmm.Communicator(rank, size)
    warmup = 5
    niters = 10
    for i in range(warmup):
        communicator.multiBcast(tensors, outputs, compute_eigen)
        communicator.synchronize()
    torch.cuda.synchronize()

    stime = time.time()
    for i in range(niters):
        communicator.multiBcast(tensors, outputs, compute_eigen)
        communicator.synchronize()
        torch.cuda.synchronize()
    etime = time.time()
    print('Avg time: ', (etime-stime)/niters)


def check():
    n = 1024
    a = torch.rand(n).float().cuda()
    a = a.view(-1, a.size(-1))
    A = a.t() @ (a)
    d, Q = torch.symeig(A, eigenvectors=True)
    print('GT shape: ', d.shape, Q.shape)
    d1, Q1 = tcmm.f_symeig(A)
    Q1 = Q1.transpose(-2, -1)
    print('Customize shape: ', d1.shape, Q1.shape)
    print('eigenvalues norm: ', (d-d1).norm(), d.norm(), d1.norm())
    print('eigenvectors norm: ', (Q-Q1).norm(), Q.norm(), Q1.norm())
    #def _goback(d, Q):
    #    back = torch.matmul(Q, torch.matmul(d.diag_embed(), Q.transpose(-2, -1)))
    #    return back
    #print('A: ', A)
    #print('d: ', d)
    #print('Q: ', Q)
    #print('bA: ', _goback(d, Q))
    #print('d1: ', d1)
    #print('Q1: ', Q1)
    #print('bA1: ', _goback(d1, Q1))

if __name__ == '__main__':
    bench()
    #bench_from_log()
    #bench_customize_comm()
    #check()
