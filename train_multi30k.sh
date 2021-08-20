#!/bin/bash
nworkers="${nworkers:-1}"
rdma="${rdma:-1}"

epochs="${epochs:-200}"
batch_size="${batch_size:-128}"
lr_mul="${lr_mul:-1}"
# lr_decay="${lr_decay:-20 35 45}"
warmup="${warmup:-4000}"

kfac="${kfac:-1}"
kfac_name="${kfac_name:-inverse}"
stat_decay="${stat_decay:-0.95}"
damping="${damping:-0.003}"
exclude_parts="${exclude_parts:-''}"

MPIPATH=/home/esetstore/.local/openmpi-4.0.1
PY=/home/esetstore/pytorch1.4/bin/python

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
    $PY examples/test_transformer.py \
        --epoch $epochs --batch-size $batch_size  --lr-mul $lr_mul --n-warmup-steps $warmup --kfac-update-freq $kfac --stat-decay $stat_decay --damping $damping --kfac-name $kfac_name --exclude-parts ${exclude_parts} \
        --data-pkl data/m30k_deen_shr.pkl --label-smoothing --proj-share-weight --embs-share-weight
