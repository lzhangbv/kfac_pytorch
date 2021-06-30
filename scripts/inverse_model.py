import math
import torch
import numpy as np
import time
import torchsso

dims = np.array(range(1, 129)) * 64
L = 10
inv_time = []

module_shape_A = [27, 1152, 1152, 1152, 1152, 1152, 1152, 1152, 1152, 1152, 1152, 1152, 1152, 1152, 1152, 1152, 1152, 1152, 1152, 1152, 2304, 2304, 2304, 2304, 2304, 2304, 2304, 2304, 2304, 2304, 2304, 2304, 2304, 2304, 2304, 2304, 2304, 2304, 4608, 4608, 4608, 4608, 4608, 4608, 4608, 4608, 4608, 4608, 4608, 4608, 4608, 4608, 4608, 4608, 4608, 513]
module_shape_G = [128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 10]

#for dim in module_shape_A:
for dim in dims:
    tensor = torch.rand((dim, dim)).cuda()
    torchsso.utils.inv(tensor)

    stime = time.time()
    for i in range(L):
        torchsso.utils.inv(tensor)
    ttime = time.time() - stime
    inv_time.append(ttime / L)

    del tensor

print('inverse time:', inv_time)

