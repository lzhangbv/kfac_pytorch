dnn=resnet56
batch_size=50
nworkers=4
rdma=1
base_lr=0.1
epochs=1

kfac_type=Femp
stat_decay=0.5
damping=0.00005
#exclude_parts=CommunicateInverse,ComputeInverse,CommunicateFactor,ComputeFactor
exclude_parts=''

# s-sgd
#epochs=$epochs base_lr=$base_lr kfac=0 dnn=$dnn nworkers=$nworkers rdma=$rdma batch_size=$batch_size ./train_cifar10.sh


# d-kfac
kfac_name=inverse_nopar
#epochs=$epochs base_lr=$base_lr kfac=1 exclude_parts=$exclude_parts damping=$damping stat_decay=$stat_decay kfac_type=$kfac_type kfac_name=$kfac_name dnn=$dnn nworkers=$nworkers rdma=$rdma batch_size=$batch_size ./train_cifar10.sh


# mpd-kfac
kfac_name=inverse
#exclude_parts=CommunicateInverse,ComputeInverse
#epochs=$epochs base_lr=$base_lr kfac=1 exclude_parts=$exclude_parts damping=$damping stat_decay=$stat_decay kfac_type=$kfac_type kfac_name=$kfac_name dnn=$dnn nworkers=$nworkers rdma=$rdma batch_size=$batch_size ./train_cifar10.sh


# mpd-kfac-rdc
kfac_name=inverse_rdc
#kfac_name=inverse_rdc_ppl
#exclude_parts=CommunicateInverse,ComputeInverse
epochs=$epochs base_lr=$base_lr kfac=1 exclude_parts=$exclude_parts damping=$damping stat_decay=$stat_decay kfac_type=$kfac_type kfac_name=$kfac_name dnn=$dnn nworkers=$nworkers rdma=$rdma batch_size=$batch_size ./train_cifar10.sh


# ds-kfac
kfac_name=inverse_nordc
#epochs=$epochs base_lr=$base_lr kfac=1 exclude_parts=$exclude_parts damping=$damping stat_decay=$stat_decay kfac_type=$kfac_type kfac_name=$kfac_name dnn=$dnn nworkers=$nworkers rdma=$rdma batch_size=$batch_size ./train_cifar10.sh

