#!/bin/bash

# model training settings
model_type="${model_type:-bert}"
batch_size="${batch_size:-4}"
eval_batch_size="${eval_batch_size:-8}"
lr="${lr:-0.00003}"
epochs="${epochs:-2}"

# kfac
# todo

params="--model_type $model_type --do_low_case --train_file /datasets/bert/dev-v1.1.json --predict_file /datasets/bert/train-v1.1.json --per_gpu_train_batch_size $batch_size --per_gpu_eval_batch_size $eval_batch_size --learning_rate $lr --num_train_epochs $epochs"

# multi-node multi-gpu settings
nworkers="${nworkers:-4}"
rdma="${rdma:-1}"

script=examples/pytorch_squad_bert.py
nworkers=$nworkers rdma=$rdma script=$script params=$params bash launch_horovod.sh