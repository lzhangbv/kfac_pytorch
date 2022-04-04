#!/bin/bash 

nworkers="${nworkers:-8}"
batch_size=4
eval_batch_size=8
weight_decay=0.0 #5e-5

##### Convergence #####
epochs=3
steps=0

# SGD
#lr=0.00002
#nworkers=$nworkers kfac=0 lr=$lr use_adamw=0 epochs=$epochs steps=$steps bash train_squad.sh
#nworkers=$nworkers kfac=0 lr=$lr use_adamw=0 epochs=6 steps=$steps bash train_squad.sh

# AdamW or Adam
lr=0.000005
#nworkers=$nworkers kfac=0 lr=$lr use_adamw=1 epochs=$epochs steps=$steps bash train_squad.sh
nworkers=$nworkers kfac=0 lr=$lr use_adamw=1 epochs=6 steps=$steps bash train_squad.sh
nworkers=$nworkers kfac=0 lr=$lr use_adamw=1 epochs=6 steps=$steps bash train_squad.sh
nworkers=$nworkers kfac=0 lr=$lr use_adamw=1 epochs=6 steps=$steps bash train_squad.sh

# K-FAC
kfac=10
fac=10
kfac_name=eigen
stat_decay=0.05
damping=0.05
#nworkers=$nworkers kfac=$kfac fac=$fac kfac_name=$kfac_name lr=$lr weight_decay=$weight_decay damping=$damping stat_decay=$stat_decay epochs=$epochs steps=$steps bash train_squad.sh

kfac_name=eigen_dp
#nworkers=$nworkers kfac=$kfac fac=$fac kfac_name=$kfac_name lr=$lr weight_decay=$weight_decay damping=$damping stat_decay=$stat_decay epochs=$epochs steps=$steps bash train_squad.sh

##### Iteration Time #####
epochs=1
kfac=1
fac=1

# SGD
#nworkers=1 kfac=0 use_adamw=0 epochs=$epochs steps=$steps bash train_squad.sh
#nworkers=$nworkers kfac=0 use_adamw=0 epochs=$epochs steps=$steps bash train_squad.sh

# KAISA
kfac_name=inverse_kaisa
#nworkers=$nworkers kfac=$kfac fac=$fac kfac_name=$kfac_name lr=$lr weight_decay=$weight_decay damping=$damping stat_decay=$stat_decay epochs=$epochs steps=$steps bash train_squad.sh
#exclude_parts=CommunicateInverse nworkers=$nworkers kfac=$kfac fac=$fac kfac_name=$kfac_name lr=$lr weight_decay=$weight_decay damping=$damping stat_decay=$stat_decay epochs=$epochs steps=$steps bash train_squad.sh
#exclude_parts=CommunicateInverse,ComputeInverse nworkers=$nworkers kfac=$kfac fac=$fac kfac_name=$kfac_name lr=$lr weight_decay=$weight_decay damping=$damping stat_decay=$stat_decay epochs=$epochs steps=$steps bash train_squad.sh
#exclude_parts=CommunicateInverse,ComputeInverse,CommunicateFactor nworkers=$nworkers kfac=$kfac fac=$fac kfac_name=$kfac_name lr=$lr weight_decay=$weight_decay damping=$damping stat_decay=$stat_decay epochs=$epochs steps=$steps bash train_squad.sh

# SPD-KFAC
kfac_name=inverse_spd
#nworkers=$nworkers kfac=$kfac fac=$fac kfac_name=$kfac_name lr=$lr weight_decay=$weight_decay damping=$damping stat_decay=$stat_decay epochs=$epochs steps=$steps bash train_squad.sh
#exclude_parts=CommunicateInverse nworkers=$nworkers kfac=$kfac fac=$fac kfac_name=$kfac_name lr=$lr weight_decay=$weight_decay damping=$damping stat_decay=$stat_decay epochs=$epochs steps=$steps bash train_squad.sh
#exclude_parts=CommunicateInverse,ComputeInverse nworkers=$nworkers kfac=$kfac fac=$fac kfac_name=$kfac_name lr=$lr weight_decay=$weight_decay damping=$damping stat_decay=$stat_decay epochs=$epochs steps=$steps bash train_squad.sh

# DP-KFAC
kfac_name=eigen_dp
#kfac_name=inverse_dp
#nworkers=$nworkers kfac=$kfac fac=$fac kfac_name=$kfac_name lr=$lr weight_decay=$weight_decay damping=$damping stat_decay=$stat_decay epochs=$epochs steps=$steps bash train_squad.sh
#exclude_parts=CommunicateInverse nworkers=$nworkers kfac=$kfac fac=$fac kfac_name=$kfac_name lr=$lr weight_decay=$weight_decay damping=$damping stat_decay=$stat_decay epochs=$epochs steps=$steps bash train_squad.sh
#exclude_parts=CommunicateInverse,ComputeInverse nworkers=$nworkers kfac=$kfac fac=$fac kfac_name=$kfac_name lr=$lr weight_decay=$weight_decay damping=$damping stat_decay=$stat_decay epochs=$epochs steps=$steps bash train_squad.sh

