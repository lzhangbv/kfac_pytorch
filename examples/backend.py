import torch
import horovod.torch as hvd
import torch.distributed as dist


class HorovodBackend:
    """
    Collective communication backend based on Horovod
    """
    def __init__(self):
        hvd.init()

    def size(self):
        return hvd.size()

    def local_rank(self):
        return hvd.local_rank()

    def rank(self):
        return hvd.rank()




class TorchBackend:
    """
    Collective communication backend based on Pytorch DDP
    """
    def __init__(self, local_rank):
        self.local_rank = local_rank
        dist.init_process_group(backend='nccl', init_method='env://')

    def size(self):
        return dist.get_world_size()

    def local_rank(self):
        return self.local_rank

    def rank(self):
        return dist.get_rank()
        
