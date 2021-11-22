#!/bin/bash
dnn="${dnn:-resnet32}"
nworkers="${nworkers:-1}"
batch_size="${batch_size:-128}"
rdma="${rdma:-1}"
kfac="${kfac:-10}"
epochs="${epochs:-100}"
base_lr="${base_lr:-0.1}"

if [ "$epochs" = "50" ]; then
lr_decay="${lr_decay:-20 35 45}"
else
lr_decay="${lr_decay:-35 75 90}"
fi

kfac_type="${kfac_type:-Femp}"
kfac_name="${kfac_name:-inverse}"
exclude_parts="${exclude_parts:-''}"
stat_decay="${stat_decay:-0.95}"
damping="${damping:-0.003}"

backend="${backend:-horovod}"

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

if [ "$backend" = "horovod" ]; then
$MPIPATH/bin/mpirun --oversubscribe --prefix $MPIPATH -np $nworkers -hostfile cluster${nworkers} -bind-to none -map-by slot \
    $params \
    $PY examples/pytorch_cifar10_resnet.py \
        --horovod 1 --base-lr $base_lr --epochs $epochs --kfac-update-freq $kfac --model $dnn --lr-decay $lr_decay --batch-size $batch_size --stat-decay $stat_decay --damping $damping  --kfac-type $kfac_type --kfac-name $kfac_name --exclude-parts ${exclude_parts}
else
$PY -m torch.distributed.launch --nproc_per_node=4 examples/pytorch_cifar10_resnet.py \
        --horovod 0 --base-lr $base_lr --epochs $epochs --kfac-update-freq $kfac --model $dnn --lr-decay $lr_decay --batch-size $batch_size --stat-decay $stat_decay --damping $damping  --kfac-type $kfac_type --kfac-name $kfac_name --exclude-parts ${exclude_parts}
fi
