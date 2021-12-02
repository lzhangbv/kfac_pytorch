#!/bin/bash

# model training settings
dnn="${dnn:-resnet32}"
batch_size="${batch_size:-50}"
base_lr="${base_lr:-0.1}"
epochs="${epochs:-1}"

if [ "$epochs" = "50" ]; then
lr_decay="${lr_decay:-20 35 45}"
else
lr_decay="${lr_decay:-35 75 90}"
fi

kfac="${kfac:-1}"
fac="${fac:-1}"
kfac_name="${kfac_name:-inverse_dp}"
exclude_parts="${exclude_parts:-''}"
stat_decay="${stat_decay:-0.95}"
damping="${damping:-0.003}"

horovod="${horovod:-0}"
params="--horovod $horovod --model $dnn --batch-size $batch_size --base-lr $base_lr --epochs $epochs --kfac-update-freq $kfac --kfac-cov-update-freq $fac --lr-decay $lr_decay --stat-decay $stat_decay --damping $damping --kfac-name $kfac_name --exclude-parts ${exclude_parts}"

# multi-node multi-gpu settings
ngpu_per_node="${ngpu_per_node:-4}"
node_count="${node_count:-1}"
node_rank="${node_rank:-1}"

nworkers="${nworkers:-4}"
rdma="${rdma:-1}"

script=examples/pytorch_cifar10_resnet.py

if [ "$horovod" = "1" ]; then
nworkers=$nworkers rdma=$rdma script=$script params=$params bash launch_horovod.sh
else
ngpu_per_node=$ngpu_per_node node_count=$node_count node_rank=$node_rank script=$script params=$params bash launch_torch.sh
fi
