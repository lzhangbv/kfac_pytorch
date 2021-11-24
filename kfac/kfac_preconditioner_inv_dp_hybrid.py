import math
import torch
import torch.optim as optim
import numpy as np
#import horovod.torch as hvd
import kfac.backend as backend  # hvd -> backend.comm

from kfac.utils import (ComputeA, ComputeG)
from kfac.utils import update_running_avg
from kfac.utils import get_block_boundary
from kfac.utils import mat_inv
from kfac.kfac_preconditioner_inv import KFAC as KFAC_INV

import logging
logger = logging.getLogger()


class KFAC(KFAC_INV):
    """
    Distributed Preconditioning Distributed K-FAC with hybrid parallelism
    Experimental: Two-level communications (block-level intra-node InverseComp + layer-level inter-node PredComm)
    Refer to: block-wise approximation refers to kfac_inv_dp_block; intra-node/inter-node communication refers to kfac_inv_kaisa
    
    Args:
      model (nn): Torch model
      lr (float): learning rate (default: 0.1)
      damping (float): Tikhonov damping parameter (default: 0.001)
      fac_update_freq (int): iterations between update KFs (default: 1)
      kfac_update_freq (int): iterations between update inverse gradient (default: 1)
      diag_blocks (int): the number of diagonal blocks used to approximate KFs (default: 4)
      kfac_batch_size (int): the subsampled batch size used to estimate the local KFs (default: 32)
      ngpu_per_node (int): the per-node number of GPUs or the gradient worker fractional size (default: 4)
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
                 diag_blocks=4,
                 kfac_batch_size=32,
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
        
        self.diag_blocks = diag_blocks
        self.kfac_batch_size = kfac_batch_size
        if backend.comm.rank() == 0:
            logger.info("diag_blocks: %s, kfac_batch_size: %s", self.diag_blocks, self.kfac_batch_size)
        
        self.ngpu_per_node = ngpu_per_node
        assert self.diag_blocks % self.ngpu_per_node == 0  # distribute diag_blocks among GPUs per-node equally
        assert backend.comm.size() % self.ngpu_per_node == 0
        
        self.intra_node_groups = [] # new distributed groups (intra-node)
        self.inter_node_groups = [] # new distributed groups (inter-node)
        self.init_comm_group()
        
        # schedule module ranks in the beginning
        self.schedule_module_ranks()
        
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


    ### Schedule KFs
    def schedule_module_ranks(self):
        """Schedule `ngpu_per_node` ranks (one node) for each module with Round-Robin"""
        if self.module_ranks is not None:
            return self.module_ranks

        module_ranks = {}
        rank_iter = 0
        for module in self.modules:
            ranks = []
            for _ in range(self.ngpu_per_node):
                rank = rank_iter % backend.comm.size()
                ranks.append(rank)
                rank_iter += 1
            module_ranks[module] = ranks

        self.module_ranks = module_ranks
        if backend.comm.rank() == 0:
            logger.info('module_ranks: %s', module_ranks.values())

    ### Compute a and g distributively (intra-node)
    def _forward_hook_event(self, module, input):
        """Hook for saving input distributively with kfac_batch_size"""
        if self.hook_enabled and torch.is_grad_enabled() and self.steps % self.fac_update_freq == 0:
            ranks = self.module_ranks[module]
            if backend.comm.rank() in ranks:
                self.m_a[module] = input[0].data[0:self.kfac_batch_size]

    def _backward_hook_event(self, module, grad_input, grad_output):
        """Hook for saving output gradient distributively with kfac_batch_size"""
        if self.hook_enabled and self.steps % self.fac_update_freq == 0:
            ranks = self.module_ranks[module]
            if backend.comm.rank() in ranks:
                self.m_g[module] = grad_output[0].data[0:self.kfac_batch_size]

    ### Compute KFs distributively (intra-node)
    def _compute_factors(self):
        """Compute As and Gs distributively"""
        for module in self.modules:
            ranks = self.module_ranks[module]
            
            if backend.comm.rank() in ranks:
                A = self.computeA(self.m_a[module], module)
                if self.steps == 0: # initialize memory as A=I
                    self.m_A[module] = torch.diag(A.new_ones(A.shape[0]))
                update_running_avg(A, self.m_A[module], self.factor_decay)

                G = self.computeG(self.m_g[module], module, batch_averaged=True)
                if self.steps == 0: # initialize memory as G=I
                    self.m_G[module] = torch.diag(G.new_ones(G.shape[0]))
                update_running_avg(G, self.m_G[module], self.factor_decay)
    
    ### Communicate KFs
    def _communicate_factors(self):
        """No KF communication"""
        pass

    ### Compute Inverse KFs distributively
    def _get_div_points(self, Ntotal, Nsections):
        """compute div_points to split Ntotal elements into Nsection blocks almost equally"""
        Neach_section, extras = divmod(Ntotal, Nsections)
        section_sizes = ([0] + extras * [Neach_section+1] + (Nsections-extras) * [Neach_section])
        return np.cumsum(section_sizes)

    def _distributed_invert_diag_blocks(self, KF, ranks):
        """invert diag block approximated matrix distributively (intra-node)"""
        Ntotal = KF.shape[0]
        Nsections = min(self.diag_blocks, Ntotal)
        div_points = self._get_div_points(Ntotal, Nsections)
        inv_blocks = []
        for i in range(Nsections):
            st = div_points[i]
            end = div_points[i + 1]
            block = KF[st:end, st:end]
            if backend.comm.rank() == ranks[i % self.ngpu_per_node]:
                inv_blocks.append(mat_inv(block))
            else:
                inv_blocks.append(block.new_zeros(block.shape))
        return inv_blocks

    def _compute_inverse(self):
        """Compute inverse factors distributively"""
        for module in self.modules:
            ranks = self.module_ranks[module]

            if backend.comm.rank() in ranks:
                A = self._add_value_to_diagonal(self.m_A[module], self.damping)
                self.m_inv_A[module] = self._distributed_invert_diag_blocks(A, ranks)

                G = self._add_value_to_diagonal(self.m_G[module], self.damping)
                self.m_inv_G[module] = self._distributed_invert_diag_blocks(G, ranks)

        # enable comm_inverse
        if self.ngpu_per_node > 1:
            self.communicate_inverse_or_not = True
        else:
            self.communicate_inverse_or_not = False
    
    ### Communicate Inverse KFs (intra-node)
    def _communicate_inverse(self):
        """Broadcast the inverse blocks to other intra-node workers"""
        handles = []

        for m in self.modules: 
            ranks = self.module_ranks[m]
            
            if backend.comm.rank() not in ranks:
                continue

            for i, inv_block in enumerate(self.m_inv_A[m]):
                local_rank = ranks[i % self.ngpu_per_node]
                group = self.intra_node_groups[local_rank]
                handles.append(backend.comm.broadcast_async_(inv_block, local_rank, group))
            
            for i, inv_block in enumerate(self.m_inv_G[m]):
                local_rank = ranks[i % self.ngpu_per_node]
                group = self.intra_node_groups[local_rank]
                handles.append(backend.comm.broadcast_async_(inv_block, local_rank, group))

        for handle in handles:
            backend.comm.synchronize(handle)

    ### Compute Preconditioned Gradients (intra-node)
    def _compute_pred(self):
        """Compute the preconditioned gradients distributively"""
        for module in self.modules: 
            ranks = self.module_ranks[module]
            
            if backend.comm.rank() in ranks:
                grad = self._get_grad(module)
                block_diag_inv_A = torch.block_diag(*self.m_inv_A[module])
                block_diag_inv_G = torch.block_diag(*self.m_inv_G[module])
                self.m_precon_grad[module] = block_diag_inv_G @ grad @ block_diag_inv_A
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
            ranks = self.module_ranks[m]
            v = self.m_precon_grad[m]
            for local_rank in ranks: # inter-node broadcast for each local gpu
                group = self.inter_node_groups[local_rank]
                
                if backend.comm.rank() not in self.get_inter_node_group(local_rank):
                    continue

                handles.append(backend.comm.broadcast_async_(v, local_rank, group))

        for handle in handles:
            backend.comm.synchronize(handle)
