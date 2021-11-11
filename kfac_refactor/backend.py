import torch
import horovod.torch as hvd
import torch.distributed as dist
import os


"""
Usuage:
    import kfac.backend as backend
    
    hvd.init() or dist.init()
    backend.init()
    backend.comm.APIs()
"""

# global comm object
comm = None

def init(backend):
    global comm
    if comm is None:
        comm = _get_comm_backend(backend)

def _get_comm_backend(backend):
        if backend == "Horovod":
            try:
                hvd.size()
                return _HorovodBackend()
            except:
                return RuntimeError('Horovod much be init before create HorovodBackend.')
        elif backend == "Torch":
            try:
                dist.get_world_size()
                return _TorchBackend()
            except:
                return RuntimeError('Torch.distributed much be init before create TorchBackend.')
        else:
            return RuntimeError('The backend is not implemented. Now only Horovod and Torch are supported.')

class _HorovodBackend:
    """
    Collective communication backend based on Horovod
    """
    def __init__(self):
        #hvd.init()
        pass

    def size(self):
        return hvd.size()

    def local_rank(self):
        return hvd.local_rank()

    def rank(self):
        return hvd.rank()


class _TorchBackend:
    """
    Collective communication backend based on Pytorch DDP
    """
    def __init__(self):
        #dist.init_process_group(backend='nccl', init_method='env://')
        pass

    def size(self):
        return dist.get_world_size()

    def local_rank(self):
        try:
            return os.environ['LOCAL_RANK']
        except:
            raise RuntimeError('LOCAL_RANK must be set in the environment when using torch.distributed')

    def rank(self):
        return dist.get_rank()
        
