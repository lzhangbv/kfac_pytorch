from __future__ import print_function
import numpy as np
import reader
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

def _fit_linear_function(x, y):
    X = np.array(x).reshape((-1, 1))
    Y = np.array(y)
    print('x: ', X)
    print('y: ', Y)
    model = LinearRegression()
    model.fit(X, Y)
    alpha = model.intercept_
    beta = model.coef_[0]
    #A = np.vstack([X, np.ones(len(X))]).T
    #beta, alpha = np.linalg.lstsq(A, Y, rcond=None)[0]
    return alpha, beta

def model_bcast_log():
    fn='logs/nccl-bcast-n16IB.log'
    #fn='logs/nccl-bcast-n64.log'
    #fn='~/LinZ/kfac_pytorch/packages/nccl-tests/nccl-allreduce-32-100GbIB-RTX2080ti.log'
    sizes, comms, errors = reader.read_times_from_nccl_log(fn, start=1024, end=1024*1024*512, original=True)
    print('sizes: ', sizes)
    print('comms: ', comms)
    print('errors: ', errors)
    alpha, beta = _fit_linear_function(np.array(sizes), comms)
    print('alpha: ', alpha, ', beta: ', beta)
    py = alpha + beta * np.array(sizes)

    fig, ax = plt.subplots(figsize=(8.,4))
    ax.plot(sizes, comms, marker='o', label='measured')
    ax.plot(sizes, py, marker='^', label='fit')
    plt.show()

def model_inverse_compute_log():
    fn = 'logs/inverse-resnet50.log'
    sizes, times = reader.read_tensorsize_vs_time(fn)
    print('sizes: ', sizes)
    print('times: ', times)
    alpha, beta = _fit_linear_function(np.array(sizes), times)
    print('alpha: ', alpha, ', beta: ', beta)
    py = alpha + beta * np.array(sizes)

    fig, ax = plt.subplots(figsize=(8.,4))
    ax.scatter(sizes, times, marker='o', label='measured')
    ax.scatter(sizes, py, marker='^', label='fit')
    plt.show()


if __name__ == '__main__':
    model_bcast_log()
    #model_inverse_compute_log()

