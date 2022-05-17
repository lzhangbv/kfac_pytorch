import itertools
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
#import torchsso
#import tcmm

Linear_Average = True

def mat_inv(x, method="cholesky"):
    if method == "inv":
        return torch.linalg.inv(x).contiguous()
    elif method == "cholesky":
        u = torch.linalg.cholesky(x)
        return torch.cholesky_inverse(u).contiguous()
    elif method == "torchsso": # cuSOLVER based inverse package
        return torchsso.utils.inv(x)
    else:
        raise NotImplementedError

def mat_eig(x, method="eigh"):
    if method == "eigh":
        eigen_val, eigen_vec = torch.linalg.eigh(x)
        return eigen_val, eigen_vec.contiguous()
    elif method == "tcmm": # cuSOLVER based sym-eig package
        eigen_val, eigen_vec = tcmm.f_symeig(x)
        return eigen_val, eigen_vec.transpose(-2, -1).contiguous()
    else:
        raise NotImplementedError


def _extract_patches(x, kernel_size, stride, padding):
    """Extract patches from convolutional layer

    Args:
      x: The input feature maps.  (batch_size, in_c, h, w)
      kernel_size: the kernel size of the conv filter (tuple of two elements)
      stride: the stride of conv operation  (tuple of two elements)
      padding: number of paddings. be a tuple of two elements
    
    Returns:
      Tensor of shape (batch_size, out_h, out_w, in_c*kh*kw)
    """
    if padding[0] + padding[1] > 0:
        x = F.pad(x, (padding[1], padding[1], padding[0],
                      padding[0])).data  # Actually check dims
    x = x.unfold(2, kernel_size[0], stride[0])
    x = x.unfold(3, kernel_size[1], stride[1])
    x = x.transpose_(1, 2).transpose_(2, 3).contiguous()
    x = x.view(
        x.size(0), x.size(1), x.size(2),
        x.size(3) * x.size(4) * x.size(5))
    return x


#def update_running_avg(new, current, alpha):
#    """Compute running average of matrix in-place
#
#    current = (1-alpha) * new + (alpha) * current
#    """
#    current *= alpha / (1 - alpha)
#    current += new
#    current *= (1 - alpha)

def update_running_avg(new, current, alpha):
    """Compute running average of matrix in-place
    current = alpha * new + (1-alpha) * current
    """
    current.mul_(1-alpha)
    current.add_(new, alpha=alpha)

class ComputeA:
    
    @classmethod
    def __call__(cls, a, layer):
        if isinstance(layer, nn.Linear):
            cov_a = cls.linear(a, layer)
        elif isinstance(layer, nn.Conv2d):
            cov_a = cls.conv2d(a, layer)
        else:
            raise NotImplementedError("KFAC does not support layer: ".format(layer))
        return cov_a

    @staticmethod
    def conv2d(a, layer):
        batch_size = a.size(0)
        a = _extract_patches(a, layer.kernel_size, layer.stride, layer.padding)
        spatial_size = a.size(1) * a.size(2)
        a = a.view(-1, a.size(-1))
        if layer.bias is not None:
            a = torch.cat([a, a.new(a.size(0), 1).fill_(1)], 1)
        a = a.div_(spatial_size)
        return a.t() @ (a / batch_size) 

    @staticmethod
    def linear(a, layer):
        if len(a.shape) > 2:
            a = torch.mean(a, 1)         # average dim of num_word
        batch_size = a.size(0)
        if layer.bias is not None:
            a = torch.cat([a, a.new(a.size(0), 1).fill_(1)], 1)
        return a.t() @ (a / batch_size)


class ComputeG:
    
    @classmethod
    def __call__(cls, g, layer, batch_averaged):
        if isinstance(layer, nn.Conv2d):
            cov_g = cls.conv2d(g, layer, batch_averaged)
        elif isinstance(layer, nn.Linear):
            cov_g = cls.linear(g, layer, batch_averaged)
        else:
            raise NotImplementedError("KFAC does not support layer: ".format(layer))
        return cov_g

    @staticmethod
    def conv2d(g, layer, batch_averaged):
        spatial_size = g.size(2) * g.size(3)
        batch_size = g.shape[0]
        g = g.transpose(1, 2).transpose(2, 3)
        g = g.reshape(-1, g.size(-1))

        if batch_averaged:
            g = g * batch_size
        g = g * spatial_size
        cov_g = g.t() @ (g / g.size(0))
        return cov_g

    @staticmethod
    def linear(g, layer, batch_averaged):
        if len(g.shape) > 2:
            g = torch.mean(g, 1)
        batch_size = g.size(0)
        if batch_averaged:
            cov_g = g.t() @ (g * batch_size)
        else:
            cov_g = g.t() @ (g / batch_size)
        return cov_g


# WIP
class _ComputeA:
    def __init__(self, linear_average=Linear_Average, conv2d_average=False, use_tensor_core=False):
        self.linear_average = linear_average
        self.conv2d_average = conv2d_average
        self.use_tensor_core = use_tensor_core

    def __call__(self, a, layer):
        """Return Kronecker Factor A"""
        batch_size, a = self.get_activation(a, layer)
        assert a.shape[-1] == self.get_dimension(a, layer)

        if self.use_tensor_core:
            return tcmm.f_gemm_ex(a.t(), a/batch_size)
        else:
            return a.t() @ (a / batch_size)

    def get_activation(self, a, layer):
        """Return batch size and activation matrix, shape: _ * dim"""
        if isinstance(layer, nn.Linear):
            batch_size = a.size(0)
            if len(a.shape) > 2:
                if self.linear_average:
                    a = torch.mean(a, list(range(len(a.shape)))[1:-1])
                else: # to be checked
                    a = a.view(-1, a.shape[-1])
                    batch_size = a.size(0)
            if layer.bias is not None:
                a = torch.cat([a, a.new(a.size(0), 1).fill_(1)], 1)
            return batch_size, a
        
        elif isinstance(layer, nn.Conv2d):
            batch_size = a.size(0)
            a = _extract_patches(a, layer.kernel_size, layer.stride, layer.padding)
            if self.conv2d_average: # to be checked
                a = torch.mean(a, [1, 2])
                if layer.bias is not None:
                    a = torch.cat([a, a.new(a.size(0), 1).fill_(1)], 1)
            else:
                spatial_size = a.size(1) * a.size(2)
                a = a.view(-1, a.size(-1))
                if layer.bias is not None:
                    a = torch.cat([a, a.new(a.size(0), 1).fill_(1)], 1)
                a.div_(spatial_size)
            return batch_size, a
        
        else:
            raise NotImplementedError("KFAC does not support layer: ".format(layer))

    def get_dimension(self, a, layer):
        """Return the dimension of Kronecker Factor A"""
        if isinstance(layer, nn.Linear):
            dim_A = layer.in_features
        elif isinstance(layer, nn.Conv2d):
            dim_A = layer.in_channels * np.prod(layer.kernel_size)
        else:
            raise NotImplementedError("KFAC does not support layer: ".format(layer))
        if layer.bias is not None:
            dim_A += 1
        return dim_A


class _ComputeG:
    def __init__(self, linear_average=Linear_Average, conv2d_average=False, use_tensor_core=False):
        self.linear_average = linear_average
        self.conv2d_average = conv2d_average
        self.use_tensor_core = use_tensor_core

    def __call__(self, g, layer, batch_averaged=True):
        """Return Kronecker Factor G"""
        batch_size, g = self.get_deviation(g, layer, batch_averaged)
        assert g.shape[-1] == self.get_dimension(g, layer)

        if self.use_tensor_core:
            return tcmm.f_gemm_ex(g.t(), g/batch_size)
        else:
            return g.t() @ (g / batch_size)

    def get_dimension(self, g, layer):
        """Return the dimension of KF G"""
        if isinstance(layer, nn.Linear):
            return layer.out_features
        elif isinstance(layer, nn.Conv2d):
            return layer.out_channels
        else:
            raise NotImplementedError("KFAC does not support layer: ".format(layer))

    def get_deviation(self, g, layer, batch_averaged=True):
        """Return the batch size and deviation w.r.t. the pre-activation output, shape: _ * dim"""
        if isinstance(layer, nn.Linear):
            batch_size = g.size(0)
            if batch_averaged:
                g = g * batch_size
            
            if len(g.shape) > 2: 
                assert len(g.shape) == 3    # batch_size * seq_len * dim
                if self.linear_average:
                    g = torch.mean(g, 1)
                else: # to be checked
                    g = g.reshape(-1, g.size(-1))
                    batch_size = g.size(0)
            return batch_size, g
        
        elif isinstance(layer, nn.Conv2d):
            batch_size = g.size(0)
            if batch_averaged:
                g = g * batch_size
            
            spatial_size = g.size(2) * g.size(3)    # batch_size * n_filters * out_h * out_w
            if self.conv2d_average:
                g = torch.mean(g, [2, 3])
            else:
                g = g.transpose(1, 2).transpose(2, 3)
                g = g.reshape(-1, g.size(-1))
                batch_size = g.size(0)
                g = g * spatial_size
            return batch_size, g

        else:
            raise NotImplementedError("KFAC does not support layer: ".format(layer))

