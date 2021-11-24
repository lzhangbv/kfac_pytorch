import math
import torch
import torch.optim as optim
import numpy as np
#import horovod.torch as hvd
import kfac.backend as backend

from kfac.utils import (ComputeA, ComputeG)
from kfac.utils import update_running_avg
from kfac.utils import mat_eig
from kfac.kfac_preconditioner_inv import KFAC as KFAC_INV

import logging

logger = logging.getLogger()


class KFAC(KFAC_INV):
    """
    Model-Parallelism Distributed K-FAC Preconditioner with implicit eigen-decomposition
    Refer to: Convolutional Neural Network Training with Distributed K-FAC (SC 2020)
    
    Args:
      model (nn): Torch model
      lr (float): learning rate (default: 0.1)
      damping (float): Tikhonov damping parameter (default: 0.001)
      fac_update_freq (int): iterations between update KFs (default: 1)
      kfac_update_freq (int): iterations between update inverse gradient (default: 1)
      distribute_layer_factor: whether distribute two KFs into different workers for eigen-decompositions
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
                 distribute_layer_factors=None, # layer-wise or factor-wise
                 kl_clip=0.001,
                 factor_decay=0.95,
                 exclude_vocabulary_size=None,
                 hook_enabled=True,
                 exclude_parts=''):
        
        super(KFAC, self).__init__(model=model, lr=lr, damping=damping, 
                                    fac_update_freq=fac_update_freq, 
                                    kfac_update_freq=kfac_update_freq,
                                    communicate_inverse_or_not=True, # force to comm_inverse
                                    kl_clip=kl_clip, 
                                    factor_decay=factor_decay,
                                    exclude_vocabulary_size=exclude_vocabulary_size,
                                    hook_enabled=hook_enabled,
                                    exclude_parts=exclude_parts)

        #self.computeA = ComputeA()
        #self.computeG = ComputeG()

        # Dictionaries keyed by `module` to store the eigen-vectors and eigen-values
        self.m_QA, self.m_QG = {}, {}
        self.m_dA, self.m_dG = {}, {}
        
        # Determine whether distribute layer factors (used for rank scheduling)
        if distribute_layer_factors is None:
            self.distribute_layer_factors = True \
                    if backend.comm.size() > len(self.modules) else False
        else:
            self.distribute_layer_factors = distribute_layer_factors


    ### Schedule KFs
    def schedule_module_ranks(self):
        """Schedule ranks for each rank/module with Round-Robin"""
        if self.module_ranks is not None:
            return self.module_ranks

        module_ranks = {}
        rank_iter = 0
        for module in self.modules:
            rank_a = rank_iter % backend.comm.size()
            if self.distribute_layer_factors:
                rank_iter += 1
                rank_g = rank_iter % backend.comm.size()
            else:
                rank_g = rank_a
            module_ranks[module] = (rank_a, rank_g)
            rank_iter += 1

        self.module_ranks = module_ranks
        if backend.comm.rank() == 0:
            logger.info('module_ranks: %s', module_ranks.values())
    

    ### Eigen-decompose KFs
    def _compute_inverse(self):
        """Eigen-decompose factors distributively"""
        for module in self.modules:
            if self.steps == 0: # initialize memory as dA=0, dG=0, QA=0, QG=0
                A = self.m_A[module]
                self.m_dA[module] = A.new_zeros(A.shape[0])
                self.m_QA[module] = A.new_zeros(A.shape)
                G = self.m_G[module]
                self.m_dG[module] = G.new_zeros(G.shape[0])
                self.m_QG[module] = G.new_zeros(G.shape)

            rank_a, rank_g = self.module_ranks[module]

            if backend.comm.rank() == rank_a:
                dA, QA = mat_eig(self.m_A[module])
                self.m_QA[module] = QA
                self.m_dA[module] = torch.mul(dA, (dA > self.eps).float())

            if backend.comm.rank() == rank_g:
                dG, QG = mat_eig(self.m_G[module])
                self.m_QG[module] = QG
                self.m_dG[module] = torch.mul(dG, (dG > self.eps).float())

    ### Communicate Inverse KFs
    def _communicate_inverse(self):
        """Broadcast the eigen-vectors and eigen-values to other workers"""
        handles = []

        for m in self.modules: 
            rank_a, rank_g = self.module_ranks[m]
            handles.append(backend.comm.broadcast_async_(self.m_QA[m], rank_a))
            handles.append(backend.comm.broadcast_async_(self.m_dA[m], rank_a))
            handles.append(backend.comm.broadcast_async_(self.m_QG[m], rank_g))
            handles.append(backend.comm.broadcast_async_(self.m_dG[m], rank_g))

        for handle in handles:
            backend.comm.synchronize(handle)

    ### Compute Preconditioned Gradients
    def _compute_pred(self):
        """Compute the preconditioned gradients"""
        for module in self.modules: 
            grad = self._get_grad(module)
            v1 = self.m_QG[module].t() @ grad @ self.m_QA[module]
            v2 = v1 / (self.m_dG[module].unsqueeze(1) * self.m_dA[module].unsqueeze(0) + self.damping)
            precon_grad = self.m_QG[module] @ v2 @ self.m_QA[module].t()
            self.m_precon_grad[module] = precon_grad
