#!/bin/bash

# model training settings
model_type="${model_type:-bert}"
batch_size="${batch_size:-4}"
eval_batch_size="${eval_batch_size:-8}"
lr="${lr:-0.00003}"
epochs="${epochs:-2}"
steps="${steps:-15000}"
#steps=200 # fast test
use_adamw="${use_adamw:-0}"
weight_decay="${weight_decay:-0.0}"

# kfac
kfac="${kfac:-0}"
fac="${fac:-0}"
kfac_name="${kfac_name:-inverse_dp}"
stat_decay="${stat_decay:-0.95}"
damping="${damping:-0.003}"

params="--model_type $model_type --do_lower_case --data_dir /datasets/bert --train_file train-v1.1.json --predict_file dev-v1.1.json --tokenizer_name_or_path /datasets/bert/tokenizer --per_gpu_train_batch_size $batch_size --per_gpu_eval_batch_size $eval_batch_size --learning_rate $lr --weight_decay $weight_decay --num_train_epochs $epochs --max_steps $steps --use-adamw $use_adamw --kfac-update-freq $kfac --kfac-cov-update-freq $fac --stat-decay $stat_decay --damping $damping --kfac-name $kfac_name"

# multi-node multi-gpu settings
nworkers="${nworkers:-4}"
rdma="${rdma:-1}"


#PY=/home/esetstore/pytorch1.8/bin/python # temp env with transformers
PY=/home/esetstore/pytorch1.10/bin/python
script=examples/pytorch_squad_bert.py
PY=$PY nworkers=$nworkers rdma=$rdma script=$script params=$params bash launch_horovod.sh
