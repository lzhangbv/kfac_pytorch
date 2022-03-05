#!/bin/bash 

nworkers="${nworkers:-8}"
batch_size=4
eval_batch_size=8
lr=0.00003
weight_decay=0.0 #5e-5

epochs=3
steps=0

# SGD
#nworkers=$nworkers kfac=0 use_adamw=0 epochs=$epochs steps=$steps bash train_squad.sh

# AdamW or Adam
#nworkers=$nworkers kfac=0 use_adamw=1 epochs=$epochs steps=$steps bash train_squad.sh

# K-FAC
kfac=100
fac=10
kfac_name=eigen
#kfac_name=inverse
stat_decay=0.05
damping=0.05
lr=0.000005

#nworkers=$nworkers kfac=$kfac fac=$fac kfac_name=$kfac_name lr=$lr weight_decay=$weight_decay damping=$damping stat_decay=$stat_decay epochs=$epochs steps=$steps bash train_squad.sh

kfac_name=eigen_dp
nworkers=$nworkers kfac=$kfac fac=$fac kfac_name=$kfac_name lr=$lr weight_decay=$weight_decay damping=$damping stat_decay=$stat_decay epochs=$epochs steps=$steps bash train_squad.sh

