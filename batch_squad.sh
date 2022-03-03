#!/bin/bash 

nworkers=4
batch_size=4
eval_batch_size=8
lr=0.00003

epochs=2
steps=200

# SGD
#kfac=0 use_adamw=0 epochs=$epochs steps=$steps bash train_squad.sh

# AdamW
#kfac=0 use_adamw=1 epochs=$epochs steps=$steps bash train_squad.sh

# K-FAC
kfac=100
fac=10
kfac_name=eigen
#kfac_name=inverse
stat_decay=0
damping=0.05

kfac=$kfac fac=$fac kfac_name=$kfac_name damping=$damping stat_decay=$stat_decay epochs=$epochs steps=$steps bash train_squad.sh
