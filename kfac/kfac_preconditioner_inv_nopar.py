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
import logging
import tcmm
import torchsso

logger = logging.getLogger()


class KFAC(optim.Optimizer):
    """KFAC Distributed Gradient Preconditioner

    Computes the natural gradient of a model in place with a layer-wise
    FIM approximation. Layer computations are distributed across workers
    using Horovod.

    Usage:
      optimizer = optim.SGD(model.parameters(), ...)
      optimizer = hvd.DistributedOptimizer(optimizer, ...)
      preconditioner = KFAC(model, ...)
      ... 
      for i, (data, target) in enumerate(train_loader):
          optimizer.zero_grad()
          output = model(data)
          loss = criterion(output, target)
          loss.backward()
          optimizer.synchronize()
          preconditioner.step()
          with optimizer.skip_synchronize():
              optimizer.step()

    Args:
      model (nn): Torch model to precondition
      lr (float, optional): learning rate (default: 0.1)
      factor_decay (float, optional): running average coefficient for Kronecker
          factors (default: 0.95)
      damping (float, optional): Tikhonov damping parameter (default: 0.001)
      kl_clip (float, optional): clipping parameter for gradient scaling
          (default: 0.001)
      fac_update_freq (int, optional): iterations between calculating and
          updating the running average of the Kronecker factors (default: 10)
      kfac_update_freq (int, optional): iterations between applying gradient
          preconditioning (default: 100)
      batch_averaged (bool, optional): boolean representing if the gradient
          is alrady averaged across the batches (default: True)
      diag_blocks (int, optional): Experimental: number of diagonal blocks to
          approximate the Kronecker factor eigendecomposition with. 
          `diag_blocks=1` computes the eigendecomposition of the entire factor
          (default: 1)
      diag_warmup (int, optional): number of epochs to wait before starting
          the block diagonal factor approximation (default: 0)
      distribute_layer_factors (bool, optional): if `True`, computes factors A
          and G on different workers else computes A and G for a single layer
          on the same worker. If `None`, determines best value based on layer
          count (default: None)
    """
    def __init__(self,
                 model,
                 lr=0.1,
                 hook_enabled=True,
                 factor_decay=0.95,
                 damping=0.001,
                 kl_clip=0.001,
                 fac_update_freq=10,
                 kfac_update_freq=100,
                 batch_averaged=True,
                 diag_blocks=1,
                 diag_warmup=0,
                 distribute_layer_factors=False,
                 sparse=False,
                 sparse_ratio=0.01,
                 exclude_parts=''):
                 #exclude_parts='CommunicateInverse,ComputeInverse,CommunicateFactor,ComputeFactor'):

        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 < factor_decay <= 1:
            raise ValueError("Invalid factor decay rate: {}".format(factor_decay))
        if not 0.0 < damping:
            raise ValueError("Invalid damping: {}".format(damping))
        if not 0.0 < kl_clip:
            raise ValueError("Invalid clipping value: {}".format(kl_clip))
        if not 0 < fac_update_freq:
            raise ValueError("Invalid factor update frequency: {}".format(fac_update_freq))
        if not 0 < kfac_update_freq:
            raise ValueError("Invalid K-FAC update frequency: {}".format(kfac_update_freq))
        if not 0 == kfac_update_freq % fac_update_freq:
            print("WARNING: it is suggested that kfac_update_freq be a multiple of fac_update_freq")
        if not 0 < diag_blocks:
            raise ValueError("Invalid diagonal block approx count: {}".format(diag_blocks))
        if not 0 <= diag_blocks:
            raise ValueError("Invalid diagonal block approx count: {}".format(diag_blocks))
        if not 1 == diag_blocks:
            print("WARNING: diag_blocks > 1 is experimental and may give poor results.")

        # For compatibility with `KFACParamScheduler`
        defaults = dict(lr=lr,
                        damping=damping,
                        fac_update_freq=fac_update_freq,
                        kfac_update_freq=kfac_update_freq) 

        super(KFAC, self).__init__(model.parameters(), defaults)

        self.computeA = ComputeA()
        self.computeG = ComputeG()
        self.known_modules = {'Linear', 'Conv2d'}
        self.modules = []
        self.module_names = []
        # register hooks for known modules
        self.hook_enabled = hook_enabled
        self._register_modules(model)

        self.steps = 0

        # Dictionaries keyed by `module` to storing the factors and inverse factors
        self.m_a, self.m_g = {}, {}
        self.m_A, self.m_G = {}, {}
        self.m_inv_A, self.m_inv_G = {}, {}
        self.module_ranks = None

        self.sparse = sparse
        self.sparse_ratio = sparse_ratio
        self.residualsA, self.residualsG = {}, {}

        self.factor_decay = factor_decay
        self.kl_clip = kl_clip
        self.fac_update_freq = fac_update_freq
        self.kfac_update_freq = kfac_update_freq
        self.diag_blocks = diag_blocks
        self.diag_warmup = diag_warmup
        self.batch_averaged = batch_averaged

        self.exclude_communicate_inverse = True if exclude_parts.find('CommunicateInverse') >=0 else False
        self.exclude_compute_inverse = True if exclude_parts.find('ComputeInverse') >=0 else False
        self.exclude_communicate_factor = True if exclude_parts.find('CommunicateFactor') >=0 else False
        self.exclude_compute_factor = True if exclude_parts.find('ComputeFactor') >=0 else False
        
        # Compute ideal value for `distribute_layer_factors` based on
        # registered module count
        if distribute_layer_factors is None:
            self.distribute_layer_factors = True \
                    if hvd.size() > len(self.modules) else False
        else:
            self.distribute_layer_factors = distribute_layer_factors

        self.eps = 1e-10  # for numerical stability
        self.rank_iter = cycle(list(range(hvd.size())))

    def set_hook_enabled(self, mode=True):
        self.hook_enabled = mode

    def _save_input(self, module, input):
        """Hook for saving layer input"""
        if self.hook_enabled and torch.is_grad_enabled() and self.steps % self.fac_update_freq == 0:
            self.m_a[module] = input[0].data

    def _save_grad_output(self, module, grad_input, grad_output):
        """Hook for saving gradient w.r.t output"""
        if self.hook_enabled and self.steps % self.fac_update_freq == 0:
            self.m_g[module] = grad_output[0].data

    def _register_modules(self, model):
        """Register hooks to all supported layers in the model"""
        name_idx = 0
        for module in model.modules():
            classname = module.__class__.__name__
            if classname in self.known_modules:
                self.modules.append(module)
                module.register_forward_pre_hook(self._save_input)
                module.register_backward_hook(self._save_grad_output)
                module_name = 'module_name_%s_%d' % (classname, name_idx)
                self.module_names.append(module_name)
                name_idx += 1

    def _init_A(self, factor, module):
        """Initialize memory for factor A and its inverse"""
        self.m_A[module] = torch.diag(factor.new_ones(factor.shape[0]))
        self.m_inv_A[module] = factor.new_zeros(factor.shape)

    def _update_module_A(self, module):
        A = self.computeA(self.m_a[module], module)
        if self.steps == 0:
            self._init_A(A, module)
        update_running_avg(A, self.m_A[module], self.factor_decay)

    def _update_A(self):
        """Compute and update factor A for all modules"""
        for module in self.modules: 
            self._update_module_A(module)
    
    def _init_G(self, factor, module):
        """Initialize memory for factor G and its eigendecomp"""
        self.m_G[module] = torch.diag(factor.new_ones(factor.shape[0]))
        self.m_inv_G[module] = factor.new_zeros(factor.shape)

    def _update_module_G(self, module):
        G = self.computeG(self.m_g[module], module, self.batch_averaged)
        if self.steps == 0:
            self._init_G(G, module)
        update_running_avg(G, self.m_G[module], self.factor_decay)

    def _update_G(self):
        """Compute and update factor G for all modules"""
        for module in self.modules:
            self._update_module_G(module)

    def _add_value_to_diagonal(self, X, value):
        return X.add_(torch.diag(X.new(X.shape[0]).fill_(value)))
    
    def _update_inverse_A(self, module):
        """Compute inverse of A for module on all workers"""
        block = self._add_value_to_diagonal(self.m_A[module], self.damping)
        self.m_inv_A[module] = torchsso.utils.inv(block)

    def _update_inverse_G(self, module):
        """Compute inverse of G for module on all  workers"""
        block = self._add_value_to_diagonal(self.m_G[module], self.damping)
        self.m_inv_G[module] = torchsso.utils.inv(block)

    def _get_grad(self, module):
        """Get formated gradient of module

        Args:
          module: module/layer to get gradient of

        Returns:
          Formatted gradient with shape [output_dim, input_dim] for module
        """
        if module.__class__.__name__ == 'Conv2d':
            # n_filters * (in_c * kw * kh)
            grad = module.weight.grad.data.view(module.weight.grad.data.size(0), -1)  
        else:
            grad = module.weight.grad.data
        if module.bias is not None:
            grad = torch.cat([grad, module.bias.grad.data.view(-1, 1)], 1)
        return grad

    def _get_preconditioned_grad(self, module, grad):
        """Precondition gradient of module
        
        Args:
          module: module to compute preconditioned gradient for
          grad: formatted gradient from `_get_grad()`

        Returns:
          preconditioned gradient with same shape as `grad`
        """
        if not self.exclude_compute_factor:
            v = self.m_inv_G[module] @ grad @ self.m_inv_A[module]
        else:
            v = grad

        if module.bias is not None:
            v = [v[:, :-1], v[:, -1:]]
            v[0] = v[0].view(module.weight.grad.data.size()) # weight
            v[1] = v[1].view(module.bias.grad.data.size())   # bias
        else:
            v = [v.view(module.weight.grad.data.size())]
        return v

    def _update_scale_grad(self, updates):
        """Update the gradients in place and scale

        Updates the gradients in-place for all modules using the preconditioned
        gradients and scales the gradients.

        Args:
          updates (dict): dict of {module: precon_grad}
        """
        vg_sum = 0
        for module in self.modules:
            v = updates[module]
            vg_sum += (v[0] * module.weight.grad.data * self.lr ** 2).sum().item()
            if module.bias is not None:
                vg_sum += (v[1] * module.bias.grad.data * self.lr ** 2).sum().item()
        if self.exclude_communicate_inverse:
            nu = 1
        else:
            nu = min(1.0, math.sqrt(self.kl_clip / abs(vg_sum)))

        for module in self.modules:
            v = updates[module]
            module.weight.grad.data.copy_(v[0])
            module.weight.grad.data.mul_(nu)
            if module.bias is not None:
                module.bias.grad.data.copy_(v[1])
                module.bias.grad.data.mul_(nu)

    def step(self, closure=None, epoch=None):
        """Perform one K-FAC step

        Note:
        - this function should always be called before `optimizer.step()`
        - gradients must be averaged across ranks before calling `step()`

        Args:
          closure: for compatibility with the base optimizer class.
              `closure` is ignored by KFAC
          epoch (int, optional): epoch to use for determining when to end
              the `diag_warmup` period. `epoch` is not necessary if not using
              `diag_warmup`
        """

        # Update params, used for compatibilty with `KFACParamScheduler`
        group = self.param_groups[0]
        self.lr = group['lr']
        self.damping = group['damping']
        self.fac_update_freq = group['fac_update_freq']
        self.kfac_update_freq = group['kfac_update_freq']

        updates = {}
        handles = []

        if self.steps % self.fac_update_freq == 0:
            if not self.exclude_compute_factor:
                self._update_A()
                self._update_G()
            if not self.exclude_communicate_factor:
                if hvd.size() > 1:
                    if self.sparse:
                        self._allgather_factors()
                    else:
                        self._allreduce_factors()


        if self.steps % self.kfac_update_freq == 0:

            for module in self.modules:
                if not self.exclude_compute_inverse:
                    self._update_inverse_A(module)
                    self._update_inverse_G(module)

        for i, module in enumerate(self.modules):
            grad = self._get_grad(module)
            precon_grad = self._get_preconditioned_grad(module, grad)
            updates[module] = precon_grad

        self._update_scale_grad(updates)

        self.steps += 1

    def _generate_module_ranks(self):
        if self.module_ranks is not None:
            return self.module_ranks
        
        self.rank_iter.reset() 
        module_ranks = {}

        for module in self.modules:
            ranks_a = self.rank_iter.next(1)
            ranks_g = self.rank_iter.next(1) if self.distribute_layer_factors else ranks_a
            module_ranks[module] = (ranks_a[0], ranks_g[0])
        
        self.module_ranks = module_ranks

        if hvd.rank() == 0:
            logger.info('module_ranks: %s', module_ranks.values())

    def _allreduce_factors(self):
        """Allreduce the factors for all layers"""
        handles = []

        for m in self.modules:
            handles.append(hvd.allreduce_async_(self.m_A[m].data, op=hvd.Average))
            handles.append(hvd.allreduce_async_(self.m_G[m].data, op=hvd.Average))

        for handle in handles:
            hvd.synchronize(handle)

    def _allgather_factors(self):
        """Allgather the factors for all layers"""
        handles = []
        def _get_value_and_idx(sparse_tensor):
            tensor = sparse_tensor.data.view(-1)
            one_indexes = tensor != 0
            indexes = one_indexes.nonzero().data.squeeze().view(-1)
            values = tensor.data[indexes] 
            return values, indexes.int()

        for i, m in enumerate(self.modules):
            module_name = self.module_names[i]

            A_values, A_indexes = _get_value_and_idx(self.m_A[m].data)
            A_value_name = module_name + '_A_value'
            A_idx_name = module_name + '_A_idx'
            h_value = allgather_async(A_values, A_value_name)
            h_idx = allgather_async(A_indexes, A_idx_name)

            G_values, G_indexes = _get_value_and_idx(self.m_G[m].data)
            G_value_name = module_name + '_G_value'
            G_idx_name = module_name + '_G_idx'
            h_value_G = allgather_async(G_values, G_value_name)
            h_idx_G = allgather_async(G_indexes, G_idx_name)
            handles.append((h_value, h_idx, h_value_G, h_idx_G))

        for i, handle in enumerate(handles):
            module_name = self.module_names[i]
            module = self.modules[i]
            m_A = self.m_A[module].view(-1)
            m_A.fill_(0.0)
            m_G = self.m_G[module].view(-1)
            m_G.fill_(0.0)

            h_value_A, h_idx_A, h_value_G, h_idx_G = handle
            A_values = hvd.synchronize(h_value_A)
            A_indexes = hvd.synchronize(h_idx_A).long()
            m_A.scatter_add_(0, A_indexes, A_values)
            m_A.div_(hvd.size())
            
            G_values = hvd.synchronize(h_value_G)
            G_indexes = hvd.synchronize(h_idx_G).long()
            m_G.scatter_add_(0, G_indexes, G_values)
            m_G.div_(hvd.size())

    def _broadcast_inverse_factors(self):
        handles = []

        for i, m in enumerate(self.modules):
            rank_a, rank_g = self.module_ranks[m]
            name = self.module_names[i]

            h = hvd.broadcast_async_(self.m_inv_A[m], rank_a, name=name+'inverseA')
            handles.append(h)
            h = hvd.broadcast_async_(self.m_inv_G[m], rank_g, name=name+'inverseG')
            handles.append(h)
    
        for handle in handles:
            hvd.synchronize(handle)
