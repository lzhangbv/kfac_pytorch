from __future__ import print_function
import numpy as np

def get_per_worker_load(N, P, weights, placement):
    load = np.zeros(P)
    for i in range(N):
        root = placement[i]
        load[root] += weights[i]
    return load

def get_optimal_block_partition(N, P, weights):
    """
    input: 
        #tasks (N), #workers (P), weights of tasks
    output: 
        placement: task assignment results, i.e., a list of worker IDs
    """
    if N <= P:
        return np.arange(N)

    # compute the optimal bottleneck by dynamic programming
    # , where B[i, j] is the optimal bottleneck given the first (j+1) tasks and (i+1) workers

    W = np.cumsum(weights)
    B = np.zeros((P, N)) 
    B[0, :] = W

    # # original version
    # for p in range(1, P): 
    #     for i in range(p, N - P + p + 1):
    #         B[p, i] = min([max(B[p-1, j], W[i] - W[j]) for j in range(p-1,i)])

    # improved version
    for p in range(1, P):
        j = p - 1
        for i in range(p, N - P + p + 1):
            if W[i] - W[j] > B[p-1, j]:
                while W[i] - W[j] > B[p-1, j]:
                    j += 1
                if j == i or W[i] - W[j-1] < B[p-1, j]: # Important: deal with the special case of j == i
                    j = j - 1
                    B[p, i] = W[i] - W[j]
                else:
                    B[p, i] = B[p-1, j]
            else:
                B[p, i] = B[p-1, j]
            j = p - 1

    bottleneck = B[P-1, N-1]
    # print(bottleneck)

    # continuous placement until the bottleneck is reached
    placement = np.zeros(N, dtype=int)
    root = 0
    load = 0
    for i in range(N):
        if load + weights[i] <= bottleneck + 1e-06: 
            placement[i] = root
            load += weights[i]
        else:
            root += 1
            placement[i] = root
            load = weights[i]

    assert root < P

    # fill in the empty workers or not ?
    if root < P - 1:
        for i in range(1, N-1):
            if placement[i-1] == placement[i] and placement[i-1] != placement[i+1]:
                root += 1
                placement[i] = root
                if root == P-1:
                    break

    return placement


if __name__ == '__main__':
    # # test 1 (random generator)
    # N = 128
    # P = 64
    # weights = np.random.rand(N)

    # placement = get_optimal_block_partition(N, P, weights)
    # print('placement:', placement)
    # print('loads:', get_per_worker_load(N, P, weights, placement))

    # test 2 (ResNet-50 factor dimension)
    N = 54 * 2
    P = 32
    module_shape_A = [147, 64, 576, 64, 64, 256, 576, 64, 256, 576, 64, 256, 1152, 128, 256, 512, 1152, 128, 512, 1152, 128, 512, 1152, 128, 512, 2304, 256, 512, 1024, 2304, 256, 1024, 2304, 256, 1024, 2304, 256, 1024, 2304, 256, 1024, 2304, 256, 1024, 4608, 512, 1024, 2048, 4608, 512, 2048, 4608, 512, 2049]
    module_shape_G = [64, 64, 64, 256, 256, 64, 64, 256, 64, 64, 256, 128, 128, 512, 512, 128, 128, 512, 128, 128, 512, 128, 128, 512, 256, 256, 1024, 1024, 256, 256, 1024, 256, 256, 1024, 256, 256, 1024, 256, 256, 1024, 256, 256, 1024, 512, 512, 2048, 2048, 512, 512, 2048, 512, 512, 2048, 1000]
    weights = np.append(module_shape_A, module_shape_G[::-1])

    placement = get_optimal_block_partition(N, P, weights)
    print('placement:', placement)
    print('loads:', get_per_worker_load(N, P, weights, placement))
