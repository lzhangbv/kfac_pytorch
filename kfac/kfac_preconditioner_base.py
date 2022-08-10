import math
import torch
import torch.optim as optim
import numpy as np
#import horovod.torch as hvd
import kfac.backend as backend

from distutils.version import LooseVersion
import logging
logger = logging.getLogger()


class KFAC(optim.Optimizer):
    """KFAC Distributed Gradient Preconditioner
    
    Usage:
    --------------------------------------------------------
    option 1: import horovod.torch as hvd
    --------------------------------------------------------
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
      ...
    --------------------------------------------------------
    
    --------------------------------------------------------
    option 2: import torch.distributed as dist
    --------------------------------------------------------
      model = torch.nn.parallel.DistributedDataParallel(...)
      optimizer = optim.SGD(model.parameters(), ...)
      preconditioner = KFAC(model, ...)
      ... 
      for i, (data, target) in enumerate(train_loader):
          optimizer.zero_grad()
          output = model(data)
          loss = criterion(output, target)
          loss.backward()
          preconditioner.step()
          optimizer.step()
      ...
    --------------------------------------------------------

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
                 communicate_inverse_or_not=True,
                 kl_clip=0.001,
                 factor_decay=0.95,
                 exclude_vocabulary_size=None,
                 hook_enabled=True,
                 exclude_parts=''):

        # For compatibility with `KFACParamScheduler`
        defaults = dict(lr=lr,
                        damping=damping,
                        fac_update_freq=fac_update_freq,
                        kfac_update_freq=kfac_update_freq) 

        super(KFAC, self).__init__(model.parameters(), defaults)

        self.fac_update_freq = fac_update_freq
        self.kfac_update_freq = kfac_update_freq
        self.communicate_inverse_or_not = communicate_inverse_or_not
        #self.kl_clip = kl_clip
        self.kl_clip = kl_clip if (kl_clip is not None and kl_clip > 0) else None
        self.factor_decay = factor_decay
        self.exclude_vocabulary_size = exclude_vocabulary_size
        self.hook_enabled = hook_enabled

        self.exclude_communicate_inverse = True if exclude_parts.find('CommunicateInverse') >=0 else False
        self.exclude_compute_inverse = True if exclude_parts.find('ComputeInverse') >=0 else False
        self.exclude_communicate_factor = True if exclude_parts.find('CommunicateFactor') >=0 else False
        self.exclude_compute_factor = True if exclude_parts.find('ComputeFactor') >=0 else False
        
        # register hooks
        self.modules = []
        self.module_names = []
        self._register_module_hooks(model)

        # dictionaries keyed by `module` to storing KFs, inverse KFs, etc
        self.m_a, self.m_g = {}, {}
        self.m_A, self.m_G = {}, {}
        self.m_inv_A, self.m_inv_G = {}, {}
        self.m_precon_grad = {}
        
        # scheduling results
        self.module_ranks = None

        self.eps = 1e-10  # for numerical stability
        self.steps = 0

    ### Register hooks
    def set_hook_enabled(self, mode=True):
        self.hook_enabled = mode

    def _forward_hook_event(self, module, input):
        """Default: hook for saving input (a)"""
        if self.hook_enabled and torch.is_grad_enabled() and self.steps % self.fac_update_freq == 0:
            self.m_a[module] = input[0].data

    def _backward_hook_event(self, module, grad_input, grad_output):
        """Default: hook for saving gradient w.r.t output (g)"""
        if self.hook_enabled and self.steps % self.fac_update_freq == 0:
            self.m_g[module] = grad_output[0].data

    def _register_module_hooks(self, model):
        """Register forard/backward hooks to supported modules"""
        supported_modules = {'Linear', 'Conv2d'}
        name_idx = 0
        for module in model.modules():
            classname = module.__class__.__name__
            if classname in supported_modules:
                if self.exclude_vocabulary_size is not None and classname == 'Linear' and module.out_features == self.exclude_vocabulary_size:
                    continue # exclude the pre-softmax linear layer in the Transformer model
                self.modules.append(module)
                module.register_forward_pre_hook(self._forward_hook_event)
                #module.register_backward_hook(self._backward_hook_event)  # used in pytorch1.4, and pytorch1.8 (full_backward_hook is not fired when its grad_input is None)
                module.register_full_backward_hook(self._backward_hook_event)  # used in pytorch1.10
                module_name = 'module_name_%s_%d' % (classname, name_idx)
                self.module_names.append(module_name)
                name_idx += 1
        if backend.comm.rank() == 0:
            logger.info("#register modules: %s", len(self.modules))

    def schedule_module_ranks(self):
        """Schedule ranks for each module to compute KFs"""
        raise NotImplementedError

    ### KFs computations and communications
    def _compute_factors(self):
        """Compute KFs."""
        raise NotImplementedError

    def _communicate_factors(self):
        """Communicate KFs."""
        raise NotImplementedError

    def _compute_inverse(self):
        """Compute inverse KFs."""
        raise NotImplementedError

    def _communicate_inverse(self):
        """Communicate inverse KFs."""
        raise NotImplementedError

    def _compute_pred(self):
        """Compute preconditioned gradients."""
        raise NotImplementedError

    def _communicate_pred(self):
        """Communicate preconditioned gradients."""
        raise NotImplementedError

    def _update_grad_in_place(self):
        """Update preconditioned gradients in place."""
        raise NotImplementedError
    
    ### Perform one K-FAC step
    def step(self, closure=None, epoch=None):
        """Perform one K-FAC step"""

        # update params, used for compatibilty with `KFACParamScheduler`
        group = self.param_groups[0]
        self.lr = group['lr']
        self.damping = group['damping']
        self.fac_update_freq = group['fac_update_freq']
        self.kfac_update_freq = group['kfac_update_freq']

        # (0) schedule module ranks
        if self.module_ranks is None:
            self.schedule_module_ranks()

        # (1) compute and communicate KFs
        if self.steps % self.fac_update_freq == 0:
            if not self.exclude_compute_factor:
                self._compute_factors()
            if not self.exclude_communicate_factor:
                if backend.comm.size() > 1:
                    self._communicate_factors()
        
        # (2) compute and/or communicate inverse KFs
        if self.steps % self.kfac_update_freq == 0:
            if not self.exclude_compute_inverse:
                self._compute_inverse()
            if not self.exclude_communicate_inverse and self.communicate_inverse_or_not:
                if backend.comm.size() > 1:
                   self._communicate_inverse()
                    
        # (3) compute and/or communicate preconditioned gradients
        if not self.exclude_compute_inverse:
            self._compute_pred()

        if not self.exclude_communicate_inverse and not self.communicate_inverse_or_not:
            if backend.comm.size() > 1:
                self._communicate_pred()
        
        # (4) update preconditioned gradients in place
        if not self.exclude_compute_inverse:
            self._update_grad_in_place()
        
        self.steps += 1

        # clear the temporal memory iteratively
        self.m_a, self.m_g = {}, {}


class KFACParamScheduler():
    """Updates KFAC hyper-parameters at each epoch

    Args:
      kfac (KFAC): wrapped KFAC preconditioner
      damping_alpha (float): multiplicative factor of the damping (default: 1)
      damping_schedule (list): list of epochs to multiply the damping by `damping_alpha` (default: None)
      update_freq_alpha (float): multiplicative factor of the KFAC update freq (default: 1)
      update_freq_schedule (list): list of epochs to multiply the KFAC update freq by `update_freq_alpha` (default: None)
      start_epoch (int): starting epoch, for use if resuming training from checkpoint (default: 0)
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

