#!/bin/bash

# usage: node_count=$node_count node_rank=$node_rank script=$script bash kill_torch.sh

# python script
script="${script:-python}"
echo "kill remote process: $script"

# cluster settings
total_host=16
hosts=('gpu1' 'gpu2' 'gpu3' 'gpu4' 'gpu5' 'gpu6' 'gpu7' 'gpu8' 'gpu9' 'gpu10' 'gpu11' 'gpu12' 'gpu13' 'gpu14' 'gpu15' 'gpu16')

# multi-node multi-gpu settings
node_count="${node_count:-0}"
node_rank="${node_rank:-1}"

node_rank=$(expr $node_rank - 1) # array index
if [ $(expr $node_rank + $node_count) -gt $total_host ] || [ $node_rank -lt 0 ]; then
    echo "Required nodes are out of the range: from gpu1 to gpu$total_host"
    exit 0
fi

i=0
while [ $i -lt $node_count ]
do
    host=${hosts[$node_rank]}
    cmd="kill -9 \$(ps aux|grep $script | awk '{print \$2}')"
    echo "$host: $cmd"
    ssh $host $cmd
    node_rank=$(expr $node_rank + 1)
    i=$(expr $i + 1)
done

