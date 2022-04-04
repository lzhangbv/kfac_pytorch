# Artifact for DP-KFAC SC 2022

Distributed K-FAC Preconditioner in PyTorch using [Horovod](https://github.com/horovod/horovod) for communication. 

The KFAC code was originally forked from Greg Pauloski's [kfac-pytorch](https://github.com/gpauloski/kfac_pytorch).
The CIFAR-10 and ImageNet-1k training scripts are modeled afer Horovod's example PyTorch training scripts. 
The Transformer training script on Multi-30k was based on Yu-Hsiang Huang's [attention-is-all-you-need-pytorch](https://github.com/jadore801120/attention-is-all-you-need-pytorch), and the BERT training script on SQuAD was based on huggingface's [run-squad](https://github.com/huggingface/transformers/blob/main/examples/legacy/question-answering/run_squad.py).  

## Install

### Requirements

PyTorch and Horovod are required to use K-FAC.

This code is validated to run with PyTorch-1.10.0, Horovod-0.21.0, CUDA-10.2, cuDNN-7.6, and NCCL-2.6.4. 

### Installation

```
$ git clone https://github.com/lzhangbv/kfac_pytorch.git
$ cd kfac_pytorch
$ pip install -r requirements.txt
```

## Usage

The K-FAC Preconditioner can be easily added to exisiting training scripts that use `horovod.DistributedOptimizer()`.

```Python
from kfac import KFAC
... 
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
```

Note that the K-FAC Preconditioner expects gradients to be averaged across workers before calling `preconditioner.step()` so we call `optimizer.synchronize()` before hand (Normally `optimizer.synchronize()` is not called until `optimizer.step()`). 

## Configure the cluster settings

Before running the scripts, please carefully configure the configuration files in the directory of `configs`.
- configs/cluster\*: configure the host files for MPI
- configs/envs.conf: configure the cluster enviroments


## Run experiments

```
$ mkdir logs
$ bash batch.sh
```

See `python examples/pytorch_{dataset}_{model}.py --help` for a full list of hyper-parameters.
Note: if `--kfac-update-freq 0`, the K-FAC Preconditioning is skipped entirely, i.e. training is just with SGD or Adam. 

<!-- ## Citation

```
@article{pauloski2020convolutional,
    title={Convolutional Neural Network Training with Distributed K-FAC},
    author={J. Gregory Pauloski and Zhao Zhang and Lei Huang and Weijia Xu and Ian T. Foster},
    year={2020},
    pages={to appear in the proceedings of SC20},
    eprint={2007.00784},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
``` -->
