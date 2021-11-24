import math
import torch
import torch.optim as optim
import numpy as np
#import horovod.torch as hvd
import kfac.backend as backend

from kfac.utils import (ComputeA, ComputeG)
from kfac.utils import update_running_avg
from kfac.utils import mat_eig
from kfac.kfac_preconditioner_inv_dp import KFAC as KFAC_INV_DP

import logging

logger = logging.getLogger()


class KFAC(KFAC_INV_DP):
    """
    Distributed Preconditioning Distributed K-FAC with implicit eigen-decomposition 
    Refer to: Scalable K-FAC Training for Deep Neural Networks with Distributed Preconditioning (AAAI 2022?)
    
    Args:
      model (nn): Torch model
      lr (float): learning rate (default: 0.1)
      damping (float): Tikhonov damping parameter (default: 0.001)
      fac_update_freq (int): iterations between update KFs (default: 1)
      kfac_update_freq (int): iterations between update inverse gradient (default: 1)
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
                 kl_clip=0.001,
                 factor_decay=0.95,
                 exclude_vocabulary_size=None,
                 hook_enabled=True,
                 exclude_parts=''):
        
        super(KFAC, self).__init__(model=model, lr=lr, damping=damping, 
                                    fac_update_freq=fac_update_freq, 
                                    kfac_update_freq=kfac_update_freq,
                                    kl_clip=kl_clip, 
                                    factor_decay=factor_decay,
                                    exclude_vocabulary_size=exclude_vocabulary_size,
                                    hook_enabled=hook_enabled,
                                    exclude_parts=exclude_parts)

        # Dictionaries keyed by `module` to store the eigen-vectors and eigen-values
        self.m_QA, self.m_QG = {}, {}
        self.m_dA, self.m_dG = {}, {}


    ### Eigen-decompose KFs distributively
    def _compute_inverse(self):
        """Eigen-decompose factors distributively"""
        for module in self.modules:
            rank_a, rank_g = self.module_ranks[module]

            if backend.comm.rank() == rank_a:
                dA, QA = mat_eig(self.m_A[module])
                self.m_QA[module] = QA
                self.m_dA[module] = torch.mul(dA, (dA > self.eps).float())

            if backend.comm.rank() == rank_g:
                dG, QG = mat_eig(self.m_G[module])
                self.m_QG[module] = QG
                self.m_dG[module] = torch.mul(dG, (dG > self.eps).float())

    ### Compute Preconditioned Gradients distributively
    def _compute_pred(self):
        """Compute the preconditioned gradients distributively"""
        assert not self.communicate_inverse_or_not # force to comm_pred
        for module in self.modules:
            rank_a, rank_g = self.module_ranks[module]
            assert rank_a == rank_g

            if backend.comm.rank() == rank_a:
                grad = self._get_grad(module)
                v1 = self.m_QG[module].t() @ grad @ self.m_QA[module]
                v2 = v1 / (self.m_dG[module].unsqueeze(1) * self.m_dA[module].unsqueeze(0) + self.damping)
                precon_grad = self.m_QG[module] @ v2 @ self.m_QA[module].t()
                self.m_precon_grad[module] = precon_grad
            elif self.steps == 0: # initialize memory on other workers for broadcast
                grad = self._get_grad(module)
                self.m_precon_grad[module] = grad.new_zeros(grad.shape) 
