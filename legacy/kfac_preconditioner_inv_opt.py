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
from kfac.utils import estimate_bcast_time, estimate_inverse_time 
from kfac.comm import MergedCommAllReduce, MergedCommBcast, MultiTensorComm
import logging
import tcmm
import torchsso

logger = logging.getLogger()

def add_value_to_diagonal(X, value):
    return X.add_(torch.diag(X.new(X.shape[0]).fill_(value)))
    #if torch.cuda.is_available():
    #    indices = torch.cuda.LongTensor([[i, i] for i in range(X.shape[0])])
    #else:
    #    indices = torch.LongTensor([[i, i] for i in range(X.shape[0])])
    #values = X.new_ones(X.shape[0]).mul(value)
    #return X.index_put(tuple(indices.t()), values, accumulate=True)

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
                 factor_decay=0.95,
                 damping=0.001,
                 kl_clip=0.001,
                 fac_update_freq=10,
                 kfac_update_freq=100,
                 batch_averaged=True,
                 diag_blocks=1,
                 diag_warmup=0,
                 distribute_layer_factors=None,
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
        self.name_module_map = {}
        self.module_name_map = {}
        self._register_modules(model)

        self.fw_merged_comm = MergedCommAllReduce(self.module_names, prefix='forward', merge=True, single_layer=False, symmetric=True, fp16=False)
        self.bw_merged_comm = MergedCommAllReduce(self.module_names, prefix='backward', merge=False, single_layer=False, symmetric=True, fp16=False)
        self.inverseA_merged_comm = MergedCommBcast(self.module_names, prefix='inverseA')
        self.inverseG_merged_comm = MergedCommBcast(self.module_names, prefix='inverseG')
        self.multi_comm = MultiTensorComm(symmetric=True, fp16=True)
        self.steps = 0

        # Dictionaries keyed by `module` to storing the factors and
        # eigendecompositions
        self.m_a, self.m_g = {}, {}
        self.m_A, self.m_G = {}, {}
        self.m_QA, self.m_QG = {}, {}
        self.m_dA_ranks = {}
        self.m_dG_ranks = {}
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

        self.have_cleared_Q = True if self.diag_warmup == 0 else False
        self.eps = 1e-10  # for numerical stability
        self.rank_iter = cycle(list(range(hvd.size())))

    def _save_input(self, module, input):
        """Hook for saving layer input"""
        if torch.is_grad_enabled() and self.steps % self.fac_update_freq == 0:
            self.m_a[module] = input[0].data


    def _compute_forward_factor(self, module, input):
        if torch.is_grad_enabled() and self.steps % self.fac_update_freq == 0:
            self.m_a[module] = input[0].data
            if not self.exclude_compute_factor:
                self._update_module_A(module)
            if not self.exclude_communicate_factor:
                if hvd.size() > 1:
                    name = self.module_name_map[module]
                    self.fw_merged_comm.allreduce_async_(name, self.m_A[module].data)

    def _save_grad_output(self, module, grad_input, grad_output):
        """Hook for saving gradient w.r.t output"""
        if self.steps % self.fac_update_freq == 0:
            self.m_g[module] = grad_output[0].data

    def _compute_backward_factor(self, module, grad_input, grad_output):
        if self.steps % self.fac_update_freq == 0:
            self.m_g[module] = grad_output[0].data
            if not self.exclude_compute_factor:
                self._update_module_G(module)
            if not self.exclude_communicate_factor:
                if hvd.size() > 1:
                    name = self.module_name_map[module]
                    self.bw_merged_comm.allreduce_async_(name, self.m_G[module].data)

    def _register_modules(self, model):
        """Register hooks to all supported layers in the model"""
        name_idx = 0
        for module in model.modules():
            classname = module.__class__.__name__
            if classname in self.known_modules:
                self.modules.append(module)

                module.register_forward_pre_hook(self._compute_forward_factor)
                module.register_backward_hook(self._compute_backward_factor)

                module_name = 'module_name_%s_%d' % (classname, name_idx)
                self.module_names.append(module_name)
                self.name_module_map[module_name] = module
                self.module_name_map[module] = module_name
                name_idx += 1

    def _init_A(self, factor, module):
        """Initialize memory for factor A and its eigendecomp"""
        self.m_A[module] = torch.diag(factor.new(factor.shape[0]).fill_(1))
        self.m_QA[module] = factor.new_zeros(factor.shape)

    def _init_G(self, factor, module):
        """Initialize memory for factor G and its eigendecomp"""
        self.m_G[module] = torch.diag(factor.new(factor.shape[0]).fill_(1))
        self.m_QG[module] = factor.new_zeros(factor.shape)

    def _clear_eigen(self):
        """Clear eigendecompositions

        Useful for when switching between `diag_blocks=1` and `diag-blocks>1`
        because eigendecompositions saved in place and the off-diagonals must
        be cleared.
        """
        for module in self.modules:
            self.m_QA[module].fill_(0)
            self.m_QG[module].fill_(0)

    def _update_module_A(self, module):
        a = self.computeA(self.m_a[module], module)
        if self.steps == 0:
            self._init_A(a, module)
        update_running_avg(a, self.m_A[module], self.factor_decay)
        if self.sparse:
            sparsification(self.m_A[module], module, ratio=self.sparse_ratio, residuals=self.residualsA)

    def _update_A(self):
        """Compute and update factor A for all modules"""
        for module in self.modules: 
            self._update_module_A(module)
            #if hvd.rank() == 0:
            #    data = self.m_A[module] #ComputeA.get_data(self.m_a[module], module)
            #    d = data.view(-1)
            #    indexes = d.nonzero().data.squeeze().view(-1)
            #    nnz = indexes.numel() 
            #    numel = d.numel()
            #    sparsity = (numel-nnz)*1.0/numel
            #    logger.info('vector A name: %s, shape: %s, sparsity: %f', module, data.shape, sparsity)

    def _update_module_G(self, module):
        g = self.computeG(self.m_g[module], module, self.batch_averaged)
            #logger.info('G Name: %s, shape: %s', module, g.shape)
        if self.steps == 0:
            self._init_G(g, module)
        update_running_avg(g, self.m_G[module], self.factor_decay)
        if self.sparse:
            sparsification(self.m_G[module], module, ratio=self.sparse_ratio, residuals=self.residualsG)

    def _update_G(self):
        """Compute and update factor G for all modules"""
        for module in self.modules:
            self._update_eigen_G(module)

            #if hvd.rank() == 0:
            #    data = self.m_G[module] #ComputeG.get_data(self.m_g[module], module, self.batch_averaged)
            #    d = data.view(-1)
            #    indexes = d.nonzero().data.squeeze().view(-1)
            #    nnz = indexes.numel() 
            #    numel = d.numel()
            #    sparsity = (numel-nnz)*1.0/numel
            #    logger.info('vector G name: %s, shape: %s, sparsity: %f', module, data.shape, sparsity)


    def _update_inverse_A(self, module, ranks):
        """Compute eigendecomposition of A for module on specified workers

        Note: all ranks will enter this function but only the ranks specified
        in `ranks` will continue to actually compute the eigendecomposition.
        All other ranks will simply zero out their buffer for the 
        eigendecomposition for the current module. This is done so we can sum
        the eigendecompositions across all ranks to communicate the results
        of locally computed eigendecompositions.

        Args:
          module: module to compute eigendecomposition of A on
          ranks: list of horovod ranks (i.e. workers) to use when computing
              the eigendecomposition.
        """
        if hvd.rank() in ranks:
            self._distributed_compute_inverse(self.m_A[module], 
                    self.m_QA[module], ranks)
        else:
            if ranks[0] == -1:
                self._local_computer_inverse(self.m_A[module], self.m_QA[module])
            else:
                self.m_QA[module].fill_(0)

    def _update_inverse_G(self, module, ranks):
        """Compute eigendecomposition of A for module on specified workers

        See `_update_eigen_A` for more info`
        """
        if hvd.rank() in ranks:
            self._distributed_compute_inverse(self.m_G[module], 
                    self.m_QG[module], ranks)
        else:
            if ranks[0] == -1:
                self._local_computer_inverse(self.m_G[module], self.m_QG[module])
            else:
                self.m_QG[module].fill_(0)

    def _distributed_compute_inverse(self, factor, inverse, ranks):
        """Computes the eigendecomposition of a factor across ranks
        
        Assigns each rank in `ranks` to enter this function to compute a
        diagonal block of `factor`. Results are written to `evectors` and
        `evalues`. If `len(ranks)==1`, then that rank computes the
        eigendecomposition of the entire `factor`.

        Args:
            factor (tensor): tensor to eigendecompose
            inverse (tensor): tensor to save inverse of `factor` to
            ranks (list): list of ranks that will enter this function
        """
        i = ranks.index(hvd.rank())
        n = len(ranks)
        if n > min(factor.shape):
            n = min(factor.shape)

        if i < n:
            start, end = get_block_boundary(i, n, factor.shape)
            block = factor[start[0]:end[0], start[1]:end[1]]
            block = add_value_to_diagonal(block, self.damping)
            inv = torchsso.utils.inv(block)
            inverse.data[start[0]:end[0], start[1]:end[1]].copy_(inv)

    def _local_computer_inverse(self, factor, inverse):
        block = factor[0:, 0:]
        block = add_value_to_diagonal(block, self.damping)
        inv = torchsso.utils.inv(block)
        inverse.data[0:, 0:].copy_(inv)

    def _get_diag_blocks(self, module, diag_blocks):
        """Helper method for determining number of diag_blocks to use

        Overrides `diag_blocks` if the `module` does not support
        `diag_blocks>1`. I.e. for a Linear layer, we do not want to
        use a `diag_blocks>1`.

        Args:
          module: module
          diag_blocks (int): default number of diag blocks to use
        """
        return diag_blocks if module.__class__.__name__ == 'Conv2d' else 1

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
        v = self.m_QG[module] @ grad @ self.m_QA[module]

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
        #print('fac_update_freq: ', self.fac_update_freq)
        #print('kfac_update_freq: ', self.kfac_update_freq)

        updates = {}
        handles = []

        if epoch is None:
            if self.diag_warmup > 0:
                print("WARNING: diag_warmup > 0 but epoch was not passed to "
                      "KFAC.step(). Defaulting to no diag_warmup")
            diag_blocks = self.diag_blocks
        else:
            diag_blocks = self.diag_blocks if epoch >= self.diag_warmup else 1

        if hvd.size() > 1 and self.steps % self.fac_update_freq == 0:
            self.fw_merged_comm.synchronize()
            self.bw_merged_comm.synchronize()

        # if we are switching from no diag approx to approx, we need to clear
        # off-block-diagonal elements
        if not self.have_cleared_Q and \
                epoch == self.diag_warmup and \
                self.steps % self.kfac_update_freq == 0:
            self._clear_eigen()
            self.have_cleared_Q = True

        if self.steps % self.kfac_update_freq == 0:
            # reset rank iter so device get the same layers
            # to compute to take advantage of caching
            self.rank_iter.reset() 
            handles = []

            #eigen_ranks = self._generate_eigen_ranks(epoch)
            eigen_ranks = self._generate_eigen_ranks_uniform(epoch)
            #eigen_ranks = self._generate_eigen_ranks_naive(epoch)
            #inverse_As = []
            #A_ranks = []
            #inverse_Gs = []
            #G_ranks = []
            rank_to_tensors = {}

            for module in self.modules:
                ranks_a, ranks_g = eigen_ranks[module]
                self.m_dA_ranks[module] = ranks_a[0]
                self.m_dG_ranks[module] = ranks_g[0]
                rank_a = ranks_a[0]
                rank_g = ranks_g[0]

                name = self.module_name_map[module]
                if not self.exclude_compute_inverse:
                    self._update_inverse_A(module, ranks_a)
                #if hvd.size() > 1 and rank_a >= 0:
                #    self.inverseA_merged_comm.bcast_async_(name, self.m_QA[module], rank_a)

                if not self.exclude_compute_inverse:
                    self._update_inverse_G(module, ranks_g)
                #if hvd.size() > 1 and rank_g >= 0:
                #    self.inverseG_merged_comm.bcast_async_(name, self.m_QG[module], rank_g)
                #if rank_a not in rank_to_tensors:
                #    rank_to_tensors[rank_a] = []
                #rank_to_tensors[rank_a].append((name, self.m_QA[module], self.m_QG[module]))
                if not self.exclude_communicate_inverse:
                    if hvd.size() > 1 and rank_g >= 0:
                        self.multi_comm.bcast_async_([name], [self.m_QA[module], self.m_QG[module]], rank_g)
            #if hvd.size() > 1:
            #    for rank in rank_to_tensors.keys():
            #        names = []
            #        tensors = []
            #        for name, ta, tb in rank_to_tensors[rank]:
            #            names.append(name)
            #            tensors.append(ta)
            #            tensors.append(tb)
            #        self.multi_comm.bcast_async_(names, tensors, rank)

        if hvd.size() > 1 and self.steps % self.kfac_update_freq == 0:
            #self.inverseA_merged_comm.synchronize()
            #self.inverseG_merged_comm.synchronize()
            self.multi_comm.synchronize()

        for i, module in enumerate(self.modules):

            grad = self._get_grad(module)
            precon_grad = self._get_preconditioned_grad(module, grad)
            updates[module] = precon_grad

        self._update_scale_grad(updates)

        self.steps += 1

    def _generate_eigen_ranks_naive(self, epoch):
        if self.module_ranks is not None:
            return self.module_ranks
        module_ranks = {}
        diag_blocks = self.diag_blocks if epoch >= self.diag_warmup else 1
        buckets = [0] * hvd.size()
        for module in self.modules:
            # Get ranks to compute this layer on
            n = self._get_diag_blocks(module, diag_blocks)
            ranks_a = self.rank_iter.next(n)
            ranks_g = ranks_a 
            #ranks_g = self.rank_iter.next(n) if self.distribute_layer_factors \
            #                                 else ranks_a
            module_ranks[module] = (ranks_a, ranks_g)
            buckets[ranks_a[0]] += self.m_A[module].shape[1]
            buckets[ranks_g[0]] += self.m_G[module].shape[1]
        self.module_ranks = module_ranks
        if hvd.rank() == 0:
            logger.info('buckets: %s', buckets)
            logger.info('module_ranks: %s', module_ranks.values())
        return module_ranks

    def _generate_eigen_ranks_uniform(self, epoch):
        if self.module_ranks is not None:
            return self.module_ranks
        module_ranks = {}
        diag_blocks = self.diag_blocks if epoch >= self.diag_warmup else 1
        buckets = [0] * hvd.size()
        dimensions = []
        module_factors = []
        for i, m in enumerate(self.modules):
            name = self.module_names[i]
            a_dimension = self.m_A[m].shape[1]
            g_dimension = self.m_G[m].shape[1]

            #if hvd.rank() == 0:
            #    logger.info('A Name: %s, shape: %s', m, self.m_A[m].shape)
            #    logger.info('G Name: %s, shape: %s', m, self.m_G[m].shape)
            dimensions.append(a_dimension)
            module_factors.append(name+'-A')
            dimensions.append(g_dimension)
            module_factors.append(name+'-G')

        descending_sorted_idx = np.argsort(dimensions)[::-1]
        A_ranks = {}
        G_ranks = {}
        bi = 0
        dimension_A = 0
        dimension_G = 0
        for i in descending_sorted_idx:
            factor = module_factors[i]
            dimension = dimensions[i]
            if factor[-1] == 'G':
                dimension_G += dimension * dimension
                continue

            m_i = self.module_names.index(factor[0:-2])
            m = self.modules[m_i]

            #bcast_time = estimate_bcast_time(dimension*dimension, hvd.size())
            #inverse_time = estimate_inverse_time(dimension)
            #if hvd.rank() == 0:
            #    print('dimension: %d, bcast_time: %f, inverse_time: %f' % (dimension, bcast_time, inverse_time))

            if dimension < 1024:
            #if inverse_time < bcast_time: 
                bi = -1
            else:
                bi = np.argmin(buckets)
                buckets[bi] += dimension
            if factor[-1] == 'A':
                dimension_A += dimension * dimension
                A_ranks[m] = (bi,)
                G_ranks[m] = (bi,)
            else:
                G_ranks[m] = (bi,)
        for m in self.modules:
            module_ranks[m] = (A_ranks[m], G_ranks[m])

        self.module_ranks = module_ranks
        if hvd.rank() == 0:
            logger.info('buckets: %s', buckets)
            logger.info('module_ranks: %s', module_ranks.values())
            logger.info('# of elementes of A: %d', dimension_A)
            logger.info('# of elementes of G: %d', dimension_G)
        return module_ranks

    def _generate_eigen_ranks(self, epoch):
        if self.module_ranks is not None:
            return self.module_ranks
        module_ranks = {}
        diag_blocks = self.diag_blocks if epoch >= self.diag_warmup else 1
        buckets = [0] * hvd.size()

        for module in self.modules:
            i = np.argmin(buckets)
            if hvd.rank() == 0:
                logger.info('A Name: %s, shape: %s', module, self.m_A[module].shape)
                logger.info('G Name: %s, shape: %s', module, self.m_G[module].shape)
            a_dimension = self.m_A[module].shape[1]
            g_dimension = self.m_G[module].shape[1]
            #buckets[i] += (a_dimension) + g_dimension)
            buckets[i] += a_dimension
            ranks_a = (i,)
            i = np.argmin(buckets)
            ranks_g = (i,)
            buckets[i] += g_dimension

            module_ranks[module] = (ranks_a, ranks_g)
        self.module_ranks = module_ranks
        if hvd.rank() == 0:
            logger.info('buckets: %s', buckets)
            logger.info('module_ranks: %s', module_ranks.values())
        return module_ranks

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

    def _allreduce_eigendecomp(self):
        """Allreduce the eigendecompositions for all layers

        Note: we use `op=hvd.Sum` to simulate an allgather`. Each rank will
        either compute the eigendecomposition for a factor or just return
        zeros so we sum instead of averaging.
        """
        handles = []

        for m in self.modules:
            handles.append(hvd.allreduce_async_(self.m_QA[m].data, op=hvd.Sum))
            handles.append(hvd.allreduce_async_(self.m_QG[m].data, op=hvd.Sum))
    
        for handle in handles:
            hvd.synchronize(handle)

    def _broadcast_eigendecomp(self):
        """Broadcasts the eigendecompositions for all layers

        Note: we use `op=hvd.Sum` to simulate an allgather`. Each rank will
        either compute the eigendecomposition for a factor or just return
        zeros so we sum instead of averaging.
        """
        handles = []
        rank = hvd.rank()

        for i, m in enumerate(self.modules):
            rank_a = self.m_dA_ranks[m]
            rank_g = self.m_dG_ranks[m]
            name = self.module_names[i]

            h = hvd.broadcast_async_(self.m_QA[m], rank_a, name=name+'mQA')
            handles.append(h)
            h = hvd.broadcast_async_(self.m_QG[m], rank_g, name=name+'mQG')
            handles.append(h)
    
        for handle in handles:
            hvd.synchronize(handle)


class KFACParamScheduler():
    """Updates KFAC parameters according to the epoch

    Similar to `torch.optim.lr_scheduler.StepLR()`

    Usage:
      Call KFACParamScheduler.step() each epoch to compute new parameter
      values.

    Args:
      kfac (KFAC): wrapped KFAC preconditioner
      damping_alpha (float, optional): multiplicative factor of the damping 
          (default: 1)
      damping_schedule (list, optional): list of epochs to update the damping
          by `damping_alpha` (default: None)
      update_freq_alpha (float, optional): multiplicative factor of the KFAC
          update freq (default: 1)
      update_freq_schedule (list, optional): list of epochs to update the KFAC
          update freq by `update_freq_alpha` (default: None)
      start_epoch (int, optional): starting epoch, for use if resuming training
          from checkpoint (default: 0)
    """
    def __init__(self,
                 kfac,
                 damping_alpha=1,
                 damping_schedule=None,
                 update_freq_alpha=1,
                 update_freq_schedule=None,
                 start_epoch=0):

        self.kfac = kfac
        params = self.kfac.param_groups[0]

        self.damping_base = params['damping']
        self.damping_alpha = damping_alpha
        self.damping_schedule = damping_schedule
        self.damping_factor_func = \
                self._get_factor_func(self.damping_schedule,
                                     self.damping_alpha)

        self.fac_update_freq_base = params['fac_update_freq']
        self.kfac_update_freq_base = params['kfac_update_freq']
        self.update_freq_alpha = update_freq_alpha
        self.update_freq_schedule = update_freq_schedule
        self.update_freq_factor_func = \
                self._get_factor_func(self.update_freq_schedule,
                                     self.update_freq_alpha)

        self.epoch = start_epoch

    def _get_factor_func(self, schedule, alpha):
        """Returns a function to compute an update factor using the epoch"""
        if schedule is not None:
            schedule.sort(reverse=True)
        else:
            schedule = []

        def factor_func(epoch):
            factor = 1.
            for e in schedule:
                if epoch >= e:
                    factor *= alpha
            return factor

        return factor_func

    def step(self, epoch=None):
        """Update KFAC parameters"""
        if epoch is not None:
            self.epoch = epoch
        else:
            self.epoch += 1

        params = self.kfac.param_groups[0]

        params['damping'] = self.damping_base * self.damping_factor_func(self.epoch)

        factor = self.update_freq_factor_func(self.epoch)
        params['fac_update_freq'] = int(self.fac_update_freq_base * factor)
        params['kfac_update_freq'] = int(self.kfac_update_freq_base * factor)
