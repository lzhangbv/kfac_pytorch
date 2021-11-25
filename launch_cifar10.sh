#!/bin/bash

# model training settings
dnn="${dnn:-resnet32}"
batch_size="${batch_size:-50}"
base_lr="${base_lr:-0.1}"
epochs="${epochs:-1}"
kfac="${kfac:-1}"

if [ "$epochs" = "50" ]; then
lr_decay="${lr_decay:-20 35 45}"
else
lr_decay="${lr_decay:-35 75 90}"
fi

kfac_name="${kfac_name:-inverse}"
exclude_parts="${exclude_parts:-''}"
stat_decay="${stat_decay:-0.95}"
damping="${damping:-0.003}"

params="--horovod 0  --model $dnn --batch-size $batch_size --base-lr $base_lr --epochs $epochs --kfac-update-freq $kfac --lr-decay $lr_decay --stat-decay $stat_decay --damping $damping --kfac-name $kfac_name --exclude-parts ${exclude_parts}"

# python env and script
PY=/home/esetstore/pytorch1.8/bin/python
directory=/home/esetstore/LinZ/kfac_pytorch

# cluster settings
total_host=16
hosts=('gpu1' 'gpu2' 'gpu3' 'gpu4' 'gpu5' 'gpu6' 'gpu7' 'gpu8' 'gpu9' 'gpu10' 'gpu11' 'gpu12' 'gpu13' 'gpu14' 'gpu15' 'gpu16')

# multi-node multi-gpu settings
ngpu_per_node="${ngpu_per_node:-4}"
node_count="${node_count:-2}"
node_rank="${node_rank:-1}"

node_rank=$(expr $node_rank - 1) # array index
if [ $(expr $node_rank + $node_count) -gt $total_host ] || [ $node_rank -lt 0 ]; then
    echo "Required nodes are out of the range: from gpu1 to gpu$total_host"
    exit 0
fi
master_host=${hosts[$node_rank]}

i=0
while [ $i -lt $node_count ]
do
    host=${hosts[$node_rank]}
    args="$PY -m torch.distributed.launch --nproc_per_node=$ngpu_per_node --nnodes=$node_count --node_rank=$i --master_addr=$master_host examples/pytorch_cifar10_resnet.py $params"
    echo "$host: $args"
    cmd="cd $directory; $args"
    ssh $host $cmd &
    node_rank=$(expr $node_rank + 1)
    i=$(expr $i + 1)
done

