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
    Distributed Preconditioning Distributed K-FAC with subsamping and block diagonal approximation 
    experimental: accelerate FactorComp with subsampled kfac_batch_size, and InverseComp with block diagonal approximation
    
    Args:
      model (nn): Torch model
      lr (float): learning rate (default: 0.1)
      damping (float): Tikhonov damping parameter (default: 0.001)
      fac_update_freq (int): iterations between update KFs (default: 1)
      kfac_update_freq (int): iterations between update inverse gradient (default: 1)
      diag_blocks (int): the number of diagonal blocks used to approximate KFs (default: 4)
      kfac_batch_size (int): the subsampled batch size used to estimate the local KFs (default: 32)
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
                 kl_clip=0.001,
                 factor_decay=0.95,
                 exclude_vocabulary_size=None,
                 hook_enabled=True,
                 exclude_parts=''):
        
        super(KFAC, self).__init__(model=model, lr=lr, damping=damping, 
                                    fac_update_freq=fac_update_freq, 
                                    kfac_update_freq=kfac_update_freq,
                                    communicate_inverse_or_not=False, # force to comm_pred
                                    kl_clip=kl_clip, 
                                    factor_decay=factor_decay,
                                    exclude_vocabulary_size=exclude_vocabulary_size,
                                    hook_enabled=hook_enabled,
                                    exclude_parts=exclude_parts)
        
        self.diag_blocks = int(diag_blocks)
        self.kfac_batch_size = kfac_batch_size
        if backend.comm.rank() == 0:
            logger.info("diag_blocks: %s, kfac_batch_size: %s", self.diag_blocks, self.kfac_batch_size)
        
        # schedule module ranks in the beginning
        self.schedule_module_ranks()


    ### Compute a and g distributively
    def _forward_hook_event(self, module, input):
        """Hook for saving input distributively with kfac_batch_size"""
        if self.hook_enabled and torch.is_grad_enabled() and self.steps % self.fac_update_freq == 0:
            rank_a, _ = self.module_ranks[module]
            if backend.comm.rank() == rank_a:
                self.m_a[module] = input[0].data[0:self.kfac_batch_size]

    def _backward_hook_event(self, module, grad_input, grad_output):
        """Hook for saving output gradient distributively with kfac_batch_size"""
        if self.hook_enabled and self.steps % self.fac_update_freq == 0:
            _, rank_g = self.module_ranks[module]
            if backend.comm.rank() == rank_g:
                self.m_g[module] = grad_output[0].data[0:self.kfac_batch_size]

    ### Compute KFs distributively
    def _compute_factors(self):
        """Compute As and Gs distributivelyi (todo: compute diagonal blocks only)"""
        for module in self.modules:
            rank_a, rank_g = self.module_ranks[module]
            
            if backend.comm.rank() == rank_a:
                A = self.computeA(self.m_a[module], module)
                if self.steps == 0: # initialize memory as A=I
                    self.m_A[module] = torch.diag(A.new_ones(A.shape[0]))
                update_running_avg(A, self.m_A[module], self.factor_decay)

            if backend.comm.rank() == rank_g:
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

    def _invert_diag_blocks(self, KF, inv_KF):
        """invert diag block approximated matrix"""
        Ntotal = KF.shape[0]
        Nsections = min(self.diag_blocks, Ntotal)
        div_points = self._get_div_points(Ntotal, Nsections)
        for i in range(Nsections):
            st = div_points[i]
            end = div_points[i + 1]
            block = KF[st:end, st:end]
            inverse = mat_inv(block)
            inv_KF.data[st:end, st:end].copy_(inverse)

    def _compute_inverse(self):
        """Compute inverse factors distributively"""
        for module in self.modules:
            rank_a, rank_g = self.module_ranks[module]
            if backend.comm.rank() == rank_a:
                if self.steps == 0: # initialize memory as inv_A=0
                    A = self.m_A[module]
                    self.m_inv_A[module] = A.new_zeros(A.shape)
                A = self._add_value_to_diagonal(self.m_A[module], self.damping)
                self._invert_diag_blocks(A, self.m_inv_A[module])

            if backend.comm.rank() == rank_g:
                if self.steps == 0: # initialize memory as inv_G=0
                    G = self.m_G[module]
                    self.m_inv_G[module] = G.new_zeros(G.shape)             
                G = self._add_value_to_diagonal(self.m_G[module], self.damping)
                self._invert_diag_blocks(G, self.m_inv_G[module])

    ### Compute Preconditioned Gradients distributively
    def _compute_pred(self):
        """Compute the preconditioned gradients distributively"""
        assert not self.communicate_inverse_or_not # force to comm_pred
        for module in self.modules: 
            rank_a, rank_g = self.module_ranks[module]
            assert rank_a == rank_g
            
            if backend.comm.rank() == rank_a:
                grad = self._get_grad(module)
                self.m_precon_grad[module] = self.m_inv_G[module] @ grad @ self.m_inv_A[module]
            elif self.steps == 0: # initialize memory on other workers for broadcast
                grad = self._get_grad(module)
                self.m_precon_grad[module] = grad.new_zeros(grad.shape) 