import math
import torch
import torch.optim as optim
import numpy as np
#import horovod.torch as hvd
import kfac.backend as backend  # hvd -> backend.comm

from kfac.utils import (ComputeA, ComputeG)
from kfac.utils import update_running_avg
from kfac.utils import mat_inv
from kfac.kfac_preconditioner_inv import KFAC as KFAC_INV

import logging
logger = logging.getLogger()


class KFAC(KFAC_INV):
    """
    Distributed K-FAC with hybrid communication, i.e., broadcast inverse KFs (intra-node) and broadcast preconditioned gradients (inter-node)
    Refer to: `KAISA: An Adaptive Second-Order Optimizer Framework for Deep Neural Networks` (SC21)
    
    Args:
      model (nn): Torch model
      lr (float): learning rate (default: 0.1)
      damping (float): Tikhonov damping parameter (default: 0.001)
      fac_update_freq (int): iterations between update KFs (default: 1)
      kfac_update_freq (int): iterations between update inverse gradient (default: 1)
      ngpu_per_node: per-node gpu number or gradient worker fractional size (default: 4)
      kl_clip (float): clipping parameter for gradient scaling
      factor_decay (float): running average coefficient for KFs
      exclude_vocabulary_size: exclude the pre-softmax linear layer in the Transformer
      hook_enabled (bool): enable the hook events to save the immediate states (a and g)
      exclude_parts='': exclude CommunicateInverse,ComputeInverse,CommunicateFactor,ComputeFactor for time breakdowns
    """
    def __init__(self,
                 model,
                 lr=0.1,
                 damping=0.001,
                 fac_update_freq=1,
                 kfac_update_freq=1,
                 ngpu_per_node=4,
                 kl_clip=0.001,
                 factor_decay=0.95,
                 exclude_vocabulary_size=None,
                 hook_enabled=True,
                 exclude_parts=''):
        
        super(KFAC, self).__init__(model=model, lr=lr, damping=damping, 
                                    fac_update_freq=fac_update_freq, 
                                    kfac_update_freq=kfac_update_freq,
                                    communicate_inverse_or_not=True, # init to comm_inverse
                                    kl_clip=kl_clip, 
                                    factor_decay=factor_decay,
                                    exclude_vocabulary_size=exclude_vocabulary_size,
                                    hook_enabled=hook_enabled,
                                    exclude_parts=exclude_parts)
        
        self.ngpu_per_node = ngpu_per_node
        assert backend.comm.size() % self.ngpu_per_node == 0
        
        self.intra_node_groups = [] # new distributed groups (intra-node)
        self.inter_node_groups = [] # new distributed groups (inter-node)
        self.init_comm_group()
        
    def get_intra_node_group(self, rank):
        """Get inter-node group from rank for communicating inverse KFs [ints]"""
        node = rank // self.ngpu_per_node  # node id
        intra_node_group = [node * self.ngpu_per_node + i for i in range(self.ngpu_per_node)]
        return intra_node_group
    
    def get_inter_node_group(self, rank):
        """Get inter-node group from rank for communicating preconditioned gradients [ints]"""
        gpu = rank % self.ngpu_per_node  # gpu id
        nnode = backend.comm.size() // self.ngpu_per_node
        inter_node_group = [j * self.ngpu_per_node + gpu for j in range(nnode)]
        return inter_node_group

    def init_comm_group(self):
        for rank in range(backend.comm.size()):
            intra_node_group = self.get_intra_node_group(rank)
            self.intra_node_groups.append(backend.comm.new_group(intra_node_group))
            
            inter_node_group = self.get_inter_node_group(rank)
            self.inter_node_groups.append(backend.comm.new_group(inter_node_group))


    ### Compute Inverse KFs distributively
    def _compute_inverse(self):
        """Compute inverse factors distributively"""
        for module in self.modules:
            rank_a, rank_g = self.module_ranks[module]
            if backend.comm.rank() == rank_a:
                A = self._add_value_to_diagonal(self.m_A[module], self.damping)
                self.m_inv_A[module] = mat_inv(A)
            elif self.steps == 0 and backend.comm.rank() in self.get_intra_node_group(rank_a):
                # initialize memory as inv_A=0 in other intra-node GPUs for broadcast
                A = self.m_A[module]
                self.m_inv_A[module] = A.new_zeros(A.shape)

            if backend.comm.rank() == rank_g:
                G = self._add_value_to_diagonal(self.m_G[module], self.damping)
                self.m_inv_G[module] = mat_inv(G)
            elif self.steps == 0 and backend.comm.rank() in self.get_intra_node_group(rank_g):
                # initialize memory as inv_G=0 in other intra-node GPUs for broadcast
                G = self.m_G[module]
                self.m_inv_G[module] = G.new_zeros(G.shape)
        
        # enable comm_inverse
        if self.ngpu_per_node > 1:
            self.communicate_inverse_or_not = True
        else:
            self.communicate_inverse_or_not = False
    
    ### Communicate Inverse KFs (intra-node)
    def _communicate_inverse(self):
        """Broadcast the inverse factors to other intra-node workers"""
        handles = []

        for m in self.modules: 
            rank_a, rank_g = self.module_ranks[m]
            group = self.intra_node_groups[rank_a]
            
            if backend.comm.rank() not in self.get_intra_node_group(rank_a):
                continue

            handles.append(backend.comm.broadcast_async_(self.m_inv_A[m], rank_a, group))
            handles.append(backend.comm.broadcast_async_(self.m_inv_G[m], rank_g, group))

        for handle in handles:
            backend.comm.synchronize(handle)


    ### Compute Preconditioned Gradients among grad_workers
    def _compute_pred(self):
        """Compute the preconditioned gradients among grad_workers"""
        for module in self.modules: 
            rank_a, rank_g = self.module_ranks[module]
            assert rank_a == rank_g

            if backend.comm.rank() in self.get_intra_node_group(rank_a):
                grad = self._get_grad(module)
                self.m_precon_grad[module] = self.m_inv_G[module] @ grad @ self.m_inv_A[module]
            elif self.steps == 0: # initialize memory on other workers for broadcast
                grad = self._get_grad(module)
                self.m_precon_grad[module] = grad.new_zeros(grad.shape) 
        
        # enable comm_pred
        if backend.comm.size() > self.ngpu_per_node:
            self.communicate_inverse_or_not = False
        else:
            self.communicate_inverse_or_not = True


    ### Communicate Preconditioned Gradients (inter-node)
    def _communicate_pred(self):
        """Broadcast the preconditioned gradients to other inter-node workers"""
        handles = []

        for m in self.modules:
            rank_a, _ = self.module_ranks[m]
            v = self.m_precon_grad[m]
            for local_rank in self.get_intra_node_group(rank_a): # inter-node broadcast for each local gpu
                group = self.inter_node_groups[local_rank]
                
                if backend.comm.rank() not in self.get_inter_node_group(local_rank):
                    continue

                handles.append(backend.comm.broadcast_async_(v, local_rank, group))

        for handle in handles:
            backend.comm.synchronize(handle)
