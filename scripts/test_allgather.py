import torch
import horovod.torch as hvd
import torch.distributed as dist
import os

def test_allgather():
    hvd.init()
    torch.cuda.set_device(hvd.local_rank())
    rank = hvd.rank()
    tensor = torch.rand(10).float().cuda()
    print('rank: ', rank, ', tensor: ', tensor)
    #handle = hvd.allgather_async(tensor)
    #tensor = hvd.synchronize(handle)
    handle = hvd.broadcast_async(tensor, 0)
    hvd.synchronize(handle)
    print('---------')
    print('rank: ', rank, ', tensor: ', tensor)

def test_process_set():
    hvd.init(process_sets="dynamic")
    even_set = hvd.add_process_set([0,2])
    odd_set = hvd.add_process_set([1,3])
    
    torch.cuda.set_device(hvd.local_rank())
    tensor = torch.rand(10).float().cuda() 

    for p in [hvd.global_process_set, even_set, odd_set]:
        print(p)


    if hvd.rank() in [0, 2]:
        hvd.allreduce_(tensor, process_set=even_set)
    if hvd.rank() in [1, 3]:
        hvd.allreduce_(tensor, process_set=odd_set)
    print(tensor)

def test_torch_ddp():
    dist.init_process_group(backend='nccl', init_method='env://')
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    tensor = torch.rand(10).float().cuda()

    print(tensor)
    dist.all_reduce(tensor) # in-place, sum
    print(tensor)

if __name__ == '__main__':
    #test_allgather()
    test_process_set()
    #test_torch_ddp()
