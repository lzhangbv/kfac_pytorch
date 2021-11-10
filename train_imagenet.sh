#!/bin/bash
#dnn="${dnn:-resnet32}"
dnn="${dnn:-resnet50}"
nworkers="${nworkers:-4}"
batch_size="${batch_size:-32}"
rdma="${rdma:-1}"
kfac="${kfac:-1}"
epochs="${epochs:-55}"
kfac_name="${kfac_name:-inverse}"
exclude_parts="${exclude_parts:-''}"
MPIPATH=/home/esetstore/.local/openmpi-4.0.1
PY=/home/esetstore/pytorch1.8/bin/python

if [ "$rdma" = "0" ]; then
params="-mca pml ob1 -mca btl ^openib \
    -mca btl_tcp_if_include 192.168.0.1/24 \
    -x NCCL_DEBUG=INFO  \
    -x NCCL_SOCKET_IFNAME=enp136s0f0,enp137s0f0 \
    -x NCCL_IB_DISABLE=1 \
    -x HOROVOD_CACHE_CAPACITY=0"
else
params="--mca pml ob1 --mca btl openib,vader,self --mca btl_openib_allow_ib 1 \
    -mca btl_tcp_if_include ib0 \
    --mca btl_openib_want_fork_support 1 \
    -x LD_LIBRARY_PATH  \
    -x NCCL_IB_DISABLE=0 \
    -x NCCL_SOCKET_IFNAME=ib0 \
    -x NCCL_DEBUG=INFO \
    -x HOROVOD_CACHE_CAPACITY=0"
fi
    #-x HOROVOD_FUSION_THRESHOLD=0 \

$MPIPATH/bin/mpirun --oversubscribe --prefix $MPIPATH -np $nworkers -hostfile cluster${nworkers} -bind-to none -map-by slot \
    $params \
    $PY examples/pytorch_imagenet_resnet.py \
          --base-lr 0.0125 --epochs $epochs --kfac-update-freq $kfac --kfac-cov-update-freq $kfac --model $dnn --kfac-name $kfac_name --exclude-parts ${exclude_parts} --batch-size $batch_size --lr-decay 25 35 40 45 50 \
          --train-dir /localdata/ILSVRC2012_dataset/train \
          --val-dir /localdata/ILSVRC2012_dataset/val
