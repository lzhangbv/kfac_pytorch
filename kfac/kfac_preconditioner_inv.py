import math
import torch
import torch.optim as optim
import numpy as np
#import horovod.torch as hvd
import kfac.backend as backend  # hvd -> backend.comm

from kfac.utils import (ComputeA, ComputeG)
from kfac.utils import update_running_avg
from kfac.utils import mat_inv
from kfac.kfac_preconditioner_base import KFAC as KFAC_BASE

import logging
logger = logging.getLogger()


class KFAC(KFAC_BASE):
    """
    Model-Parallelism Distributed K-FAC Preconditioner with explicit factor inversion 
    Refer to: Large-scale distributed second-order optimization using kronecker-factored approximate curvature for deep convolutional neural networks (CVPR 2019)
    
    Args:
      model (nn): Torch model
      lr (float): learning rate (default: 0.1)
      damping (float): Tikhonov damping parameter (default: 0.001)
      fac_update_freq (int): iterations between update KFs (default: 1)
      kfac_update_freq (int): iterations between update inverse gradient (default: 1)
      communicate_inverse_or_not (bool): choose to communicate inverse KFs or communicate preconditioned gradients (default: False)
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
                 communicate_inverse_or_not=False,
                 kl_clip=0.001,
                 factor_decay=0.95,
                 exclude_vocabulary_size=None,
                 hook_enabled=True,
                 exclude_parts=''):
        
        super(KFAC, self).__init__(model=model, lr=lr, damping=damping, 
                                    fac_update_freq=fac_update_freq, 
                                    kfac_update_freq=kfac_update_freq,
                                    communicate_inverse_or_not=communicate_inverse_or_not,
                                    kl_clip=kl_clip, 
                                    factor_decay=factor_decay,
                                    exclude_vocabulary_size=exclude_vocabulary_size,
                                    hook_enabled=hook_enabled,
                                    exclude_parts=exclude_parts)

        self.computeA = ComputeA()
        self.computeG = ComputeG()

    ### Schedule KFs
    def schedule_module_ranks(self):
        """Schedule ranks for each module with Round-Robin"""
        if self.module_ranks is not None:
            return self.module_ranks

        module_ranks = {}
        rank_iter = 0
        for module in self.modules:
            rank_a = rank_iter % backend.comm.size()
            rank_g = rank_a
            module_ranks[module] = (rank_a, rank_g)
            rank_iter += 1

        self.module_ranks = module_ranks
        if backend.comm.rank() == 0:
            logger.info('module_ranks: %s', module_ranks.values())
    
    ### Compute KFs
    def _compute_factors(self):
        """Compute As and Gs, and store results to m_A and m_G"""
        for module in self.modules:
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
        """All-reduce the factors"""
        handles = []
        
        for m in self.modules:
            handles.append(backend.comm.allreduce_async_(self.m_A[m], op=backend.comm.Average))
            handles.append(backend.comm.allreduce_async_(self.m_G[m], op=backend.comm.Average))
        
        for handle in handles:
            backend.comm.synchronize(handle)

    ### Compute Inverse KFs
    def _add_value_to_diagonal(self, X, value):
        return X.add_(torch.diag(X.new(X.shape[0]).fill_(value)))

    def _compute_inverse(self):
        """Compute inverse factors distributively"""
        for module in self.modules:
            if self.steps == 0: # initialize memory as inv_A=0, inv_G=0
                A = self.m_A[module]
                self.m_inv_A[module] = A.new_zeros(A.shape)
                G = self.m_G[module]
                self.m_inv_G[module] = G.new_zeros(G.shape)
            
            rank_a, rank_g = self.module_ranks[module]
            A = self.m_A[module]
            G = self.m_G[module]
            pi = torch.sqrt((A.trace()/A.shape[0])/(G.trace()/G.shape[0]))

            if backend.comm.rank() == rank_a:
                A = self._add_value_to_diagonal(self.m_A[module], (self.damping**0.5) * pi)
                self.m_inv_A[module] = mat_inv(A)

            if backend.comm.rank() == rank_g:
                G = self._add_value_to_diagonal(self.m_G[module], (self.damping**0.5) / pi)
                self.m_inv_G[module] = mat_inv(G)

    ### Communicate Inverse KFs
    def _communicate_inverse(self):
        """Broadcast the inverse factors to other workers"""
        handles = []

        for m in self.modules: 
            rank_a, rank_g = self.module_ranks[m]
            handles.append(backend.comm.broadcast_async_(self.m_inv_A[m], rank_a))
            handles.append(backend.comm.broadcast_async_(self.m_inv_G[m], rank_g))

        for handle in handles:
            backend.comm.synchronize(handle)

    ### Compute Preconditioned Gradients
    def _get_grad(self, module):
        """Get gradient with shape [output_dim, input_dim] for module"""
        if module.__class__.__name__ == 'Conv2d':
            # n_filters * (in_c * kw * kh)
            grad = module.weight.grad.data.view(module.weight.grad.data.size(0), -1)
        else:
            grad = module.weight.grad.data
        if module.bias is not None:
            grad = torch.cat([grad, module.bias.grad.data.view(-1, 1)], 1)
        return grad
    
    def _compute_pred(self):
        """Compute the preconditioned gradients"""
        for module in self.modules: 
            grad = self._get_grad(module)
            precon_grad = self.m_inv_G[module] @ grad @ self.m_inv_A[module]
            self.m_precon_grad[module] = precon_grad

    ### Communicate Preconditioned Gradients
    def _communicate_pred(self):
        """Broadcast the preconditioned gradients to other workers"""
        handles = []

        for m in self.modules:
            rank_a, rank_g = self.module_ranks[m] 
            assert rank_a == rank_g
            v = self.m_precon_grad[m]
            handles.append(backend.comm.broadcast_async_(v, rank_a))

        for handle in handles:
            backend.comm.synchronize(handle)

    ### Update preconditioned gradients in place
    def _reshape_preconditioned_grad(self, module, v):
        """Return preconditioned gradient with same shape as grad"""
        if module.bias is not None:
            v = [v[:, :-1], v[:, -1:]]
            v[0] = v[0].view(module.weight.grad.data.size()) # weight
            v[1] = v[1].view(module.bias.grad.data.size())   # bias
        else:
            v = [v.view(module.weight.grad.data.size())]
        return v

    def _update_grad_in_place(self):
        """Update the preconditioned gradients in place"""
        vg_sum = 0
        for module in self.modules:
            # reshape the preconditioned gradients
            precon_grad = self.m_precon_grad[module]
            v = self._reshape_preconditioned_grad(module, precon_grad)
            
            # accumulate vg_sum
            if self.kl_clip is not None:
                vg_sum += (v[0] * module.weight.grad.data * self.lr ** 2).sum().item()
                if module.bias is not None:
                    vg_sum += (v[1] * module.bias.grad.data * self.lr ** 2).sum().item()
            
            # copy the preconditioned gradients
            module.weight.grad.data.copy_(v[0])
            if module.bias is not None:
                module.bias.grad.data.copy_(v[1])
        
        # kl_clip
        if self.kl_clip is not None:
            if self.exclude_communicate_inverse:
                nu = 1
            else:
                nu = min(1.0, math.sqrt(self.kl_clip / abs(vg_sum)))
        
            for module in self.modules:
                module.weight.grad.data.mul_(nu)
                if module.bias is not None:
                    module.bias.grad.data.mul_(nu)

