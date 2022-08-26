import math
import torch
import numpy as np
import time


def mat_inv(x):
    u = torch.linalg.cholesky(x)
    return torch.cholesky_inverse(u)

def mat_eig(x):
    eigen_val, eigen_vec = torch.linalg.eigh(x)
    return eigen_val, eigen_vec.contiguous()

dims = np.array(range(1, 129)) * 64
L = 10
inv_time = []

module_shape_A = [27, 1152, 1152, 1152, 1152, 1152, 1152, 1152, 1152, 1152, 1152, 1152, 1152, 1152, 1152, 1152, 1152, 1152, 1152, 1152, 2304, 2304, 2304, 2304, 2304, 2304, 2304, 2304, 2304, 2304, 2304, 2304, 2304, 2304, 2304, 2304, 2304, 2304, 4608, 4608, 4608, 4608, 4608, 4608, 4608, 4608, 4608, 4608, 4608, 4608, 4608, 4608, 4608, 4608, 4608, 513]
module_shape_G = [128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 10]

#for dim in module_shape_A:
for dim in dims:
    #tensor = torch.rand((dim, dim)).cuda()
    tensor = torch.rand((dim, 128)).cuda()
    tensor = tensor @ tensor.T
    tensor.add_(torch.diag(tensor.new(dim).fill_(0.03)))
    
    #mat_inv(tensor)
    mat_eig(tensor)

    stime = time.time()
    for i in range(L):
        #mat_inv(tensor)
        mat_eig(tensor)
    ttime = time.time() - stime
    inv_time.append(ttime / L)

    del tensor

print('inverse time:', inv_time)

