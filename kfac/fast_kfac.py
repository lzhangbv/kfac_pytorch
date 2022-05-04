import math
import torch
import torch.optim as optim
import numpy as np
#import horovod.torch as hvd
import kfac.backend as backend

from kfac.utils import get_activation, get_deviation
from distutils.version import LooseVersion
import logging
logger = logging.getLogger()


class KFAC(optim.Optimizer):
    """Accelerate Distributed K-FAC with Sublinear Memory Cost
    Args:
      model (nn): Torch model
      lr (float): learning rate (default: 0.1)
      damping (float): Tikhonov damping parameter (default: 0.03)
      kl_clip (float): clipping parameter for gradient scaling
      factor_decay (float): running average coefficient for KFs
      exclude_vocabulary_size: exclude the pre-softmax linear layer in the Transformer
      hook_enabled (bool): enable the hook events to save the immediate states (a and g)
      exclude_parts='': exclude CommunicateInverse,ComputeInverse,CommunicateFactor,ComputeFactor for time breakdowns
    """
    def __init__(self,
                 model,
                 lr=0.1,
                 damping=0.03,
                 fac_update_freq=1,
                 kfac_update_freq=1,
                 kfac_batch_size=16,
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
        self.kfac_batch_size = kfac_batch_size
        self.communicate_inverse_or_not = communicate_inverse_or_not
        self.kl_clip = kl_clip if kl_clip > 0 else None
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
        
        # scheduling results
        self.module_ranks = None

        self.steps = 0

    ### Register hooks
    def set_hook_enabled(self, mode=True):
        self.hook_enabled = mode

    def _forward_hook_event(self, module, input):
        """Default: hook for saving input (a)"""
        if self.hook_enabled and torch.is_grad_enabled() and self.steps % self.fac_update_freq == 0:
            with torch.no_grad():
                new = get_activation(input[0].data[0:self.kfac_batch_size], module)
                if module not in self.m_a:
                    self.m_a[module] = new
                else:
                    self.m_a[module].mul_(self.factor_decay).add_(new, alpha=1-self.factor_decay)

    def _backward_hook_event(self, module, grad_input, grad_output):
        """Default: hook for saving gradient w.r.t output (g)"""
        if self.hook_enabled and self.steps % self.fac_update_freq == 0:
            with torch.no_grad():
                new = get_deviation(grad_output[0].data[0:self.kfac_batch_size], module)
                if module not in self.m_g:
                    self.m_g[module] = new
                else:
                    self.m_g[module].mul_(self.factor_decay).add_(new, alpha=1-self.factor_decay)

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

    ### KFs computations and communications
    def _communicate_factors(self):
        """All-reduce the m_a and m_g instead. """
        # Note: all-reduce after update running average
        handles = []
        for m in self.modules:
            handles.append(backend.comm.allreduce_async_(self.m_a[m], op=backend.comm.Average))
            handles.append(backend.comm.allreduce_async_(self.m_g[m], op=backend.comm.Average))

        for handle in handles:
            backend.comm.synchronize(handle)


	### Precondition gradients
    def _precondition_grads(self):
        """Compute preconditioned gradients."""
        vg_sum = 0
        for module in self.modules:
            # compute damped inverse KF
            a = self.m_a[module]
            inv_A = (torch.diag(a.new_ones(a.shape)) - (a.unsqueeze(1) * a.unsqueeze(0)).div_(self.damping + a @ a.T)).div_(self.damping)
            g = self.m_g[module]
            inv_G = (torch.diag(g.new_ones(g.shape)) - (g.unsqueeze(1) * g.unsqueeze(0)).div_(self.damping + g @ g.T)).div_(self.damping)

            # compute preconditioned grad
            grad = self._get_grad(module)
            precon_grad = inv_G @ grad @ inv_A
            del inv_A
            del inv_G

            # weight and bias
            if module.bias is not None:
                v = [precon_grad[:, :-1], precon_grad[:, -1:]]
                v[0] = v[0].view(module.weight.grad.data.size()) # weight
                v[1] = v[1].view(module.bias.grad.data.size())   # bias
            else:
                v = [precon_grad.view(module.weight.grad.data.size())]

            # copy the preconditioned gradients
            module.weight.grad.data.copy_(v[0])
            if module.bias is not None:
                module.bias.grad.data.copy_(v[1])

            # accumulate vg_sum (to be checked: how to do clip for preconditioned gradients)
            if self.kl_clip is not None:
                vg_sum += (v[0] * module.weight.grad.data * self.lr ** 2).sum().item()
                if module.bias is not None:
                    vg_sum += (v[1] * module.bias.grad.data * self.lr ** 2).sum().item()
            del precon_grad

        # kl clip
        if self.kl_clip is not None:
            nu = min(1.0, math.sqrt(self.kl_clip / abs(vg_sum)))

        for module in self.modules:
            module.weight.grad.data.mul_(nu)
            if module.bias is not None:
                module.bias.grad.data.mul_(nu)

    def _precondition_grads_fast(self):
        """Compute preconditioned gradients via matrix-vector production."""
        vg_sum = 0
        for module in self.modules:
            # get ma, mg, grad
            ma = self.m_a[module].view(-1, 1)
            mg = self.m_g[module].view(-1, 1)
            grad = self._get_grad(module)
            
            # compute intermediate states
            a = (ma.T @ ma).item() + self.damping
            g = (mg.T @ mg).item() + self.damping
            ag = (mg.T @ grad @ ma).item()

            v_a = (grad @ ma) @ ma.T
            v_g = mg @ (mg.T @ grad)
            v_ag = mg @ ma.T

            # compute preconditioned grads
            grad.sub_(v_a, alpha=1/a)
            grad.sub_(v_g, alpha=1/g)
            grad.add_(v_ag, alpha=ag/a/g)
            grad.div_(self.damping * self.damping)

            del v_a
            del v_g
            del v_ag

            # weight and bias
            if module.bias is not None:
                weight = grad[:, :-1].view(module.weight.grad.data.size())
                bias = grad[:, -1:].view(module.bias.grad.data.size())
                # copy the preconditioned parameters
                module.weight.grad.data.copy_(weight)
                module.bias.grad.data.copy_(bias)
                del grad

            # accumulate vg_sum (to be checked: how to do clip for preconditioned gradients)
            if self.kl_clip is not None:
                vg_sum += (module.weight.grad.data * module.weight.grad.data * self.lr ** 2).sum().item()
                if module.bias is not None:
                    vg_sum += (module.bias.grad.data * module.bias.grad.data * self.lr ** 2).sum().item()

        # todo cmp grad_clip with kl_clip
        if self.kl_clip is not None:
            nu = min(1.0, math.sqrt(self.kl_clip / abs(vg_sum)))

            for module in self.modules:
                module.weight.grad.data.mul_(nu)
                if module.bias is not None:
                    module.bias.grad.data.mul_(nu)

    def _precondition_grads_faster(self):
        """Compute preconditioned gradients via damping FIM"""
        vg_sum = 0
        for module in self.modules:
            # get ma, mg, grad
            ma = self.m_a[module].view(-1, 1)
            mg = self.m_g[module].view(-1, 1)
            grad = self._get_grad(module)
            
            # compute intermediate states
            a = (ma.T @ ma).item()
            g = (mg.T @ mg).item()
            ag = (mg.T @ grad @ ma).item()

            # compute preconditioned grads
            v = (mg @ ma.T).mul_(-ag/(a * g + self.damping))
            v.add_(grad)
            v.div_(self.damping)

            # weight and bias
            if module.bias is not None:
                weight = v[:, :-1].view(module.weight.grad.data.size())
                bias = v[:, -1:].view(module.bias.grad.data.size())
                # kl clip: grad and precon grad
                if self.kl_clip is not None:
                    vg_sum += (weight * module.weight.grad.data * self.lr ** 2).sum().item()
                    vg_sum += (bias * module.bias.grad.data * self.lr ** 2).sum().item()
                # copy
                module.weight.grad.data.copy_(weight)
                module.bias.grad.data.copy_(bias)
                del grad
            else:
                weight = v.view(module.weight.grad.data.size())
                if self.kl_clip is not None:
                    vg_sum += (weight * module.weight.grad.data * self.lr ** 2).sum().item()
                module.weight.grad.data.copy_(weight)
            del v

        # kl clip
        if self.kl_clip is not None:
            nu = min(1.0, math.sqrt(self.kl_clip / abs(vg_sum)))

            for module in self.modules:
                module.weight.grad.data.mul_(nu)
                if module.bias is not None:
                    module.bias.grad.data.mul_(nu)

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


    ### Perform one K-FAC step
    @torch.no_grad()
    def step(self, closure=None, epoch=None):
        """Perform one K-FAC step"""

        # update params, used for compatibilty with `KFACParamScheduler`
        group = self.param_groups[0]
        self.lr = group['lr']
        self.damping = group['damping']
        self.fac_update_freq = group['fac_update_freq']
        self.kfac_update_freq = group['kfac_update_freq']

        if self.steps % self.fac_update_freq == 0 and backend.comm.size() > 1:
        	self._communicate_factors()
        
        #self._precondition_grads()
        #self._precondition_grads_fast()
        self._precondition_grads_faster()

        self.steps += 1

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

