# Scalable K-FAC Training with Distributed Preconditioning

Distributed K-FAC Preconditioner in [PyTorch](https://github.com/pytorch/pytorch) and [Horovod](https://github.com/horovod/horovod). 

The K-FAC code was originally forked from Greg Pauloski's [kfac-pytorch](https://github.com/gpauloski/kfac_pytorch).
The CIFAR-10 and ImageNet-1k training scripts were borrowed from Horovod's example training scripts. 
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
$ HOROVOD_GPU_OPERATIONS=NCCL pip install horovod
```

If pip installation failed, please try to upgrade pip via `pip install --upgrade pip`. If Horovod installation with NCCL failed, please check the installation [guide](https://horovod.readthedocs.io/en/stable/install_include.html). 

## Usage

The DP_KFAC Preconditioner can be easily added to exisiting training scripts that use `horovod.DistributedOptimizer()`.

```Python
from kfac import DP_KFAC
... 
optimizer = optim.SGD(model.parameters(), ...)
optimizer = hvd.DistributedOptimizer(optimizer, ...)
preconditioner = DP_KFAC(model, inv_type='eigen', ...)
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

For the convenience of experiments, we support choosing D-KFAC variants using kfac.get_kfac_module. 

```Python
import kfac
...
KFAC = kfac.get_kfac_module(kfac='eigen_dp')
preconditioner = KFAC(model, ...)
...
```

Note that the 'eigen_dp' and 'inverse_dp' represent DP_KFAC algorithms with eigen-decomposition and matrix-inversion computations in preconditioning, respectively, while the 'eigen' and 'inverse' represent original MPD_KFAC algorithms proposed in [CVPR 2019](https://arxiv.org/abs/1811.12019) and [SC 20](https://arxiv.org/abs/2007.00784). 

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

Make sure the datasets were prepared in correct  dirs (e.g., /datasets/cifar10) before running the experiments. We downloaded Cifar-10, Cifar-100, and Imagenet datasets via Torchvision's [Datasets](https://pytorch.org/vision/stable/datasets.html), preprocessed Multi-30k dataset with torchtext and spacy (see [usage](https://github.com/jadore801120/attention-is-all-you-need-pytorch)), and preprocessed SQuAD dataset with huggingface's [SquadV1Processor](https://huggingface.co/docs/transformers/main_classes/processors). 

## Citation

```
@article{zhang2022scalable,
  title={Scalable K-FAC Training for Deep Neural Networks With Distributed Preconditioning},
  author={Zhang, Lin and Shi, Shaohuai and Wang, Wei and Li, Bo},
  journal={IEEE Transactions on Cloud Computing},
  year={2022},
  publisher={IEEE}
}
```
