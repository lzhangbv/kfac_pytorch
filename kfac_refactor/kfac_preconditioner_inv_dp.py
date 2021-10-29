import math
import torch
import torch.optim as optim
import horovod.torch as hvd
import numpy as np
from horovod.torch.mpi_ops import allgather_async

from kfac.utils import (ComputeA, ComputeG)
from kfac.utils import update_running_avg
from kfac.utils import try_contiguous
from kfac.utils import cycle
from kfac.utils import get_block_boundary
from kfac.utils import sparsification

from kfac_refactor.kfac_preconditioner_inv import KFAC as KFAC_INV

import logging
import tcmm
import torchsso

logger = logging.getLogger()


class KFAC(KFAC_INV):
    """
    Distributed Preconditioning Distributed K-FAC with explicit factor inversion 
    Refer to: Scalable K-FAC Training for Deep Neural Networks with Distributed Preconditioning (AAAI 2022?)
    
    Args:
      model (nn): Torch model
      lr (float): learning rate (default: 0.1)
      damping (float): Tikhonov damping parameter (default: 0.001)
      fac_update_freq (int): iterations between update KFs (default: 1)
      kfac_update_freq (int): iterations between update inverse gradient (default: 1)
      communicate_inverse_or_not (bool): choose to communicate inverse KFs or communicate preconditioned gradients
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
                 # communicate_inverse_or_not=False,
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
        
        # schedule module ranks in the beginning
        self.schedule_module_ranks()

    ### Compute a and g distributively
    def _forward_hook_event(self, module, input):
        """Hook for saving input distributively"""
        if self.hook_enabled and torch.is_grad_enabled() and self.steps % self.fac_update_freq == 0:
            rank_a, _ = self.module_ranks[module]
            if hvd.rank() == rank_a:
                self.m_a[module] = input[0].data

    def _backward_hook_event(self, module, grad_input, grad_output):
        """Hook for saving output gradient distributively"""
        if self.hook_enabled and self.steps % self.fac_update_freq == 0:
            _, rank_g = self.module_ranks[module]
            if hvd.rank() == rank_g:
                self.m_g[module] = grad_output[0].data

    ### Compute KFs distributively
    def _compute_factors(self):
        """Compute As and Gs distributively"""
        for module in self.modules:
            rank_a, rank_g = self.module_ranks[module]
            
            if hvd.rank() == rank_a:
                A = self.computeA(self.m_a[module], module)
                if self.steps == 0: # initialize memory as A=I
                    self.m_A[module] = torch.diag(A.new_ones(A.shape[0]))
                update_running_avg(A, self.m_A[module], self.factor_decay)

            if hvd.rank() == rank_g:
                G = self.computeG(self.m_g[module], module, batch_averaged=True)
                if self.steps == 0: # initialize memory as G=I
                    self.m_G[module] = torch.diag(G.new_ones(G.shape[0]))
                update_running_avg(G, self.m_G[module], self.factor_decay)
    
    ### Communicate KFs
    def _communicate_factors(self):
        """No KF communication"""
        pass

    ### Compute Inverse KFs distributively
    def _compute_inverse(self):
        """Compute inverse factors distributively"""
        for module in self.modules:
            rank_a, rank_g = self.module_ranks[module]
            if hvd.rank() == rank_a:
                # if self.steps == 0: # initialize memory as inv_A=0
                #     A = self.m_A[module]
                #     self.m_inv_A[module] = A.new_zeros(A.shape)
                A = self._add_value_to_diagonal(self.m_A[module], self.damping)
                self.m_inv_A[module] = torchsso.utils.inv(A)

            if hvd.rank() == rank_g:
                # if self.steps == 0: # initialize memory as inv_G=0
                #     G = self.m_G[module]
                #     self.m_inv_G[module] = G.new_zeros(G.shape)             
                G = self._add_value_to_diagonal(self.m_G[module], self.damping)
                self.m_inv_G[module] = torchsso.utils.inv(G)

    ### Compute Preconditioned Gradients distributively
    def _compute_pred(self):
        """Compute the preconditioned gradients distributively"""
        assert not self.communicate_inverse_or_not # force to comm_pred
        for module in self.modules: 
            rank_a, rank_g = self.module_ranks[module]
            assert rank_a == rank_g
            
            if hvd.rank() == rank_a:
                grad = self._get_grad(module)
                self.m_precon_grad[module] = self.m_inv_G[module] @ grad @ self.m_inv_A[module]
            elif self.steps == 0: # initialize memory on other workers for broadcast
                grad = self._get_grad(module)
                self.m_precon_grad[module] = grad.new_zeros(grad.shape) 
