import math
import torch
import torch.optim as optim
import horovod.torch as hvd
import numpy as np
from horovod.torch.mpi_ops import allgather_async

from legacy.utils import (ComputeA, ComputeG)
from legacy.utils import update_running_avg
from legacy.utils import try_contiguous
from legacy.utils import cycle
from legacy.utils import get_block_boundary
from legacy.utils import sparsification
import logging
import tcmm
import time
import torchsso

logger = logging.getLogger()

estimate_time_A = [0.00013175010681152344, 0.0011579513549804688, 0.0011622905731201172, 0.001163339614868164, 0.0011631011962890624, 0.0011394977569580077, 0.0008266210556030273, 0.000829005241394043, 0.0008294343948364258, 0.0008281707763671875, 0.0008249759674072265, 0.0008289337158203125, 0.0008284330368041992, 0.0008333921432495117, 0.0008373737335205078, 0.0008400678634643555, 0.0008365631103515625, 0.0008355617523193359, 0.000834512710571289, 0.0008332252502441407, 0.006051778793334961, 0.006056976318359375, 0.006056952476501465, 0.006049537658691406, 0.006057143211364746, 0.006056356430053711, 0.006053018569946289, 0.006051158905029297, 0.006050491333007812, 0.006055474281311035, 0.006048965454101563, 0.006051397323608399, 0.006054568290710449, 0.0060559272766113285, 0.006066560745239258, 0.006073403358459473, 0.006061959266662598, 0.006053304672241211, 0.03182971477508545, 0.03203625679016113, 0.032034444808959964, 0.03211054801940918, 0.032068943977355956, 0.032073044776916505, 0.03207738399505615, 0.032068395614624025, 0.03203463554382324, 0.03205530643463135, 0.03205585479736328, 0.032103443145751955, 0.03206741809844971, 0.032056117057800294, 0.032047080993652347, 0.032123994827270505, 0.03212919235229492, 0.0003176212310791016]
estimate_time_G = [0.00015666484832763672, 0.00014612674713134765, 0.00014829635620117188, 0.00016701221466064453, 0.00023555755615234375, 0.00023458003997802734, 0.00023586750030517577, 0.00023624897003173828, 0.00014681816101074218, 0.00014879703521728516, 0.00014846324920654298, 0.00014934539794921874, 0.0001502513885498047, 0.00014581680297851563, 0.00014772415161132813, 0.00014641284942626954, 0.0001462697982788086, 0.00014600753784179687, 0.00014696121215820312, 0.00018224716186523437, 0.000179290771484375, 0.0001822948455810547, 0.0001821279525756836, 0.00017876625061035155, 0.00017430782318115235, 0.0001788616180419922, 0.00018439292907714843, 0.00016841888427734374, 0.00018928050994873046, 0.00018115043640136718, 0.00017838478088378907, 0.0001840353012084961, 0.00021533966064453126, 0.0001862049102783203, 0.0001873493194580078, 0.00019392967224121093, 0.00018782615661621093, 0.0002820253372192383, 0.0002731800079345703, 0.00027272701263427737, 0.0002585411071777344, 0.000267481803894043, 0.0002699851989746094, 0.00027697086334228517, 0.0002799272537231445, 0.00028808116912841796, 0.00027093887329101565, 0.0002554655075073242, 0.00030405521392822265, 0.00027341842651367186, 0.0002665519714355469, 0.00025577545166015624, 0.00025708675384521483, 0.0002652406692504883, 0.0002630710601806641, 0.00010921955108642579] 


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
        # register hooks for known modules
        self.hook_enabled = hook_enabled
        self._register_modules(model)

        # tcmm communicator
        self.communicator = tcmm.Communicator(hvd.rank(), hvd.size(), 1)

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
    
    def _update_inverse_A(self, module, rank):
        """Compute inverse of A for module on specified worker"""
        if hvd.rank() == rank:
            block = self._add_value_to_diagonal(self.m_A[module], self.damping)
            self.m_inv_A[module] = torchsso.utils.inv(block)

    def _update_inverse_G(self, module, rank):
        """Compute inverse of G for module on specified worker"""
        if hvd.rank() == rank:
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
        v = self.m_inv_G[module] @ grad @ self.m_inv_A[module]

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
        
        # assign factors on workers to compute inverse
        self._generate_module_ranks()
        #self._generate_uniform_ranks()
        
        if self.steps % self.fac_update_freq == 0:
            if not self.exclude_communicate_factor:
                if hvd.size() > 1:
                    #self._reduce_symmetric_factors()
                    self._reduce_factors()
                    #self._allreduce_factors()

        if self.steps % self.kfac_update_freq == 0:
            stime = time.time()
            for i, module in enumerate(self.modules):
                rank_a, rank_g = self.module_ranks[module]

                if not self.exclude_compute_inverse:
                    self._update_inverse_A(module, rank_a)
                    self._update_inverse_G(module, rank_g)
            #logger.info("Step: inverse comp time %s on worker %s", time.time()-stime, hvd.rank())
            
            if not self.exclude_communicate_inverse:
                if hvd.size() > 1:
                    self._broadcast_inverse_factors()
            #logger.info("Step: inverse comp+comm time %s on worker %s", time.time()-stime, hvd.rank())

        for i, module in enumerate(self.modules):
            grad = self._get_grad(module)
            if not self.exclude_compute_factor:
                precon_grad = self._get_preconditioned_grad(module, grad)
            else:
                precon_grad = grad
            updates[module] = precon_grad

        self._update_scale_grad(updates)

        self.steps += 1

    def _get_factor_shape(self):
        shape_A = []
        shape_G = []
        for module in self.modules:
            if module.__class__.__name__ == 'Linear':
                dim_A = module.in_features
                dim_G = module.out_features
            elif module.__class__.__name__ == 'Conv2d':
                dim_A = module.in_channels * np.prod(module.kernel_size)
                dim_G = module.out_channels
            if module.bias is not None:
                dim_A += 1
            shape_A.append(dim_A)
            shape_G.append(dim_G)
        return shape_A, shape_G
        
    def _generate_module_ranks(self):
        if self.module_ranks is not None:
            return self.module_ranks
        
        self.rank_iter.reset()
        curr_rank = 0
        module_ranks = {}

        buckets = [0] * hvd.size()
        shape_A = [self.m_A[module].shape[1] for module in self.modules]
        shape_G = [self.m_G[module].shape[1] for module in self.modules]
        # shape_A, shape_G = self._get_factor_shape()
        if hvd.rank() == 0:
            logger.info('module_shape of A:%s', shape_A)
            logger.info('module_shape of G:%s', shape_G)

        assigned_rank = 0
        for i, module in enumerate(self.modules):
            ranks_a = self.rank_iter.next(1)
            #ranks_g = self.rank_iter.next(1)
            ranks_g = self.rank_iter.next(1) if self.distribute_layer_factors else ranks_a
            
            # debug: three-layer a group
            #if i > 0 and i % 14 == 0:
            #    assigned_rank += 1
            #    assigned_rank %= hvd.size()
            #ranks_a = (assigned_rank, )
            #ranks_g = (assigned_rank, )

            module_ranks[module] = (ranks_a[0], ranks_g[0])
            buckets[ranks_a[0]] += shape_A[i]
            buckets[ranks_g[0]] += shape_G[i]
        
        self.module_ranks = module_ranks

        if hvd.rank() == 0:
            logger.info('module_ranks: %s', module_ranks.values())
            logger.info('buckets: %s', buckets)
    
    def _generate_uniform_ranks(self):
        if self.module_ranks is not None:
            return self.module_ranks

        module_ranks = {}
        buckets = [0] * hvd.size()
        dimensions = []
        module_factors = []
        for i, m in enumerate(self.modules):
            name = self.module_names[i]
            a_dimension = self.m_A[m].shape[1]
            g_dimension = self.m_G[m].shape[1]
            #a_dimension = estimate_time_A[i]
            #g_dimension = estimate_time_G[i]

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
        
        for i in descending_sorted_idx:
            factor = module_factors[i]
            dimension = dimensions[i]
            
            m_i = self.module_names.index(factor[0:-2])
            m = self.modules[m_i]

            bi = np.argmin(buckets)
            load = dimension * dimension # square
            buckets[bi] += load
            
            if factor[-1] == 'A':
                A_ranks[m] = bi
            else:
                G_ranks[m] = bi

        for m in self.modules:
            module_ranks[m] = (A_ranks[m], G_ranks[m])

        self.module_ranks = module_ranks

        if hvd.rank() == 0:
            logger.info('module_ranks: %s', module_ranks.values())
            logger.info('buckets: %s', buckets)

        return module_ranks    
    
    def _triu_vectorization(self, tensor):
        triu_ind = torch.triu_indices(tensor.shape[0], tensor.shape[1])
        triu_vector = tensor[triu_ind[0], triu_ind[1]]
        return triu_ind, triu_vector

    def _reduce_symmetric_factors(self):
        for m in self.modules:
            rank_a, rank_g = self.module_ranks[m]
            # vectorization
            triu_ind_a, triu_vector_a = self._triu_vectorization(self.m_A[m].data)
            triu_ind_g, triu_vector_g = self._triu_vectorization(self.m_G[m].data)
            # reduce
            self.communicator.reduce(triu_vector_a, rank_a)
            self.communicator.reduce(triu_vector_g, rank_g)
            self.communicator.synchronize()
            # recovery
            if hvd.rank() == rank_a:
                triu_vector_a.div_(hvd.size())
                triu_vector_g.div_(hvd.size())
                self.m_A[m][triu_ind_a[0], triu_ind_a[1]] = triu_vector_a
                self.m_A[m][triu_ind_a[1], triu_ind_a[0]] = triu_vector_a
                self.m_G[m][triu_ind_g[0], triu_ind_g[1]] = triu_vector_g
                self.m_G[m][triu_ind_g[1], triu_ind_g[0]] = triu_vector_g


    def _reduce_factors(self):
        #raise NotImplementedError("Reduce op is not implemented by Horovod.")
        
        for m in self.modules:
            rank_a, rank_g = self.module_ranks[m]
            self.communicator.reduce(self.m_A[m].data, rank_a)
            self.communicator.reduce(self.m_G[m].data, rank_g)
        self.communicator.synchronize()
        for m in self.modules:
            rank_a, rank_g = self.module_ranks[m]
            if hvd.rank() == rank_a:
                self.m_A[m] = self.m_A[m].data / hvd.size()
            if hvd.rank() == rank_g:
                self.m_G[m] = self.m_G[m].data / hvd.size()

    def _allreduce_factors(self):
        """Allreduce the factors for all layers"""
        #handles = []

        #for m in self.modules:
        #    handles.append(hvd.allreduce_async_(self.m_A[m].data, op=hvd.Average))
        #    handles.append(hvd.allreduce_async_(self.m_G[m].data, op=hvd.Average))

        #for handle in handles:
        #    hvd.synchronize(handle)
        
        for m in self.modules:
            self.communicator.allReduce(self.m_A[m].data)
            self.communicator.allReduce(self.m_G[m].data)
        self.communicator.synchronize()
        for m in self.modules:
            self.m_A[m] = self.m_A[m].data / hvd.size()
            self.m_G[m] = self.m_G[m].data / hvd.size()

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

