#!/bin/bash

# model training settings
dnn="${dnn:-resnet50}"
batch_size="${batch_size:-32}"
base_lr="${base_lr:-0.0125}" 
epochs="${epochs:-55}"
lr_decay="${lr_decay:-25 35 40 45 50}"

kfac="${kfac:-1}"
fac="${fac:-1}"
kfac_name="${kfac_name:-inverse}"
exclude_parts="${exclude_parts:-''}"
stat_decay="${stat_decay:-0.95}"
damping="${damping:-0.003}"

horovod="${horovod:-0}"
params="--horovod $horovod --model $dnn --base-lr $base_lr --epochs $epochs --lr-decay $lr_decay --kfac-update-freq $kfac --kfac-cov-update-freq $fac --kfac-name $kfac_name --stat-decay $stat_decay --damping $damping --exclude-parts ${exclude_parts} --batch-size $batch_size --train-dir /localdata/ILSVRC2012_dataset/train --val-dir /localdata/ILSVRC2012_dataset/val"

# multi-node multi-gpu settings
ngpu_per_node="${ngpu_per_node:-4}"
node_count="${node_count:-1}"
node_rank="${node_rank:-1}"

nworkers="${nworkers:-4}"
rdma="${rdma:-1}"

script=examples/pytorch_imagenet_resnet.py

if [ "$horovod" = "1" ]; then
nworkers=$nworkers rdma=$rdma script=$script params=$params bash launch_horovod.sh
else
ngpu_per_node=$ngpu_per_node node_count=$node_count node_rank=$node_rank script=$script params=$params bash launch_torch.sh
fi
