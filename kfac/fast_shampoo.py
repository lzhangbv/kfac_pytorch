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
    """Experimental: Accelerate Distributed Shampoo
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

    def _register_module_hooks(self, model):
        """Register supported modules"""
        supported_modules = {'Linear', 'Conv2d'}
        name_idx = 0
        for module in model.modules():
            classname = module.__class__.__name__
            if classname in supported_modules:
                if self.exclude_vocabulary_size is not None and classname == 'Linear' and module.out_features == self.exclude_vocabulary_size:
                    continue # exclude the pre-softmax linear layer in the Transformer model
                self.modules.append(module)
                module_name = 'module_name_%s_%d' % (classname, name_idx)
                self.module_names.append(module_name)
                name_idx += 1
        if backend.comm.rank() == 0:
            logger.info("#register modules: %s", len(self.modules))


    def _precondition_grads(self):
        """Compute preconditioned gradients with shampoo, i.e., L^-1 G R^-1."""
        # however, in shampoo, the exact update formula is L^-1/4 G R^-1/4. 
        vg_sum = 0
        for module in self.modules:
            # get grad: dim_out * dim_in
            grad = self._get_grad(module)
            g_new = torch.mean(grad, dim=1).view(-1, 1)
            a_new = torch.mean(grad, dim=0).view(-1, 1)

            if module not in self.m_a:
                self.m_a[module] = a_new
            else:
                self.m_a[module].mul_(self.factor_decay).add_(a_new, alpha=1-self.factor_decay)
            if module not in self.m_g:
                self.m_g[module] = g_new
            else:
                self.m_g[module].mul_(self.factor_decay).add_(g_new, alpha=1-self.factor_decay)

            ma = self.m_a[module]
            mg = self.m_g[module]

            
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


    ### Perform one Shampoo step
    @torch.no_grad()
    def step(self, closure=None, epoch=None):
        """Perform one Shampoo step"""

        # update params, used for compatibilty with `KFACParamScheduler`
        group = self.param_groups[0]
        self.lr = group['lr']
        self.damping = group['damping']
        self.fac_update_freq = group['fac_update_freq']
        self.kfac_update_freq = group['kfac_update_freq']

        self._precondition_grads()
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

