dnn=resnet56
batch_size=50
nworkers=16
rdma=1
base_lr=0.1
epochs=2

kfac_type=Femp
stat_decay=0.5
damping=0.00005

# s-sgd
epochs=$epochs base_lr=$base_lr kfac=0 dnn=$dnn nworkers=$nworkers rdma=$rdma batch_size=$batch_size ./cifar10.sh


# d-kfac
kfac_name=inverse_nopar

exclude_parts=''
epochs=$epochs base_lr=$base_lr kfac=1 exclude_parts=$exclude_parts damping=$damping stat_decay=$stat_decay kfac_type=$kfac_type kfac_name=$kfac_name dnn=$dnn nworkers=$nworkers rdma=$rdma batch_size=$batch_size ./cifar10.sh

exclude_parts=CommunicateInverse,ComputeInverse
epochs=$epochs base_lr=$base_lr kfac=1 exclude_parts=$exclude_parts damping=$damping stat_decay=$stat_decay kfac_type=$kfac_type kfac_name=$kfac_name dnn=$dnn nworkers=$nworkers rdma=$rdma batch_size=$batch_size ./cifar10.sh

exclude_parts=CommunicateInverse,ComputeInverse,CommunicateFactor
epochs=$epochs base_lr=$base_lr kfac=1 exclude_parts=$exclude_parts damping=$damping stat_decay=$stat_decay kfac_type=$kfac_type kfac_name=$kfac_name dnn=$dnn nworkers=$nworkers rdma=$rdma batch_size=$batch_size ./cifar10.sh

exclude_parts=CommunicateInverse,ComputeInverse,CommunicateFactor,ComputeFactor
epochs=$epochs base_lr=$base_lr kfac=1 exclude_parts=$exclude_parts damping=$damping stat_decay=$stat_decay kfac_type=$kfac_type kfac_name=$kfac_name dnn=$dnn nworkers=$nworkers rdma=$rdma batch_size=$batch_size ./cifar10.sh


# mpd-kfac
kfac_name=inverse

exclude_parts=''
epochs=$epochs base_lr=$base_lr kfac=1 exclude_parts=$exclude_parts damping=$damping stat_decay=$stat_decay kfac_type=$kfac_type kfac_name=$kfac_name dnn=$dnn nworkers=$nworkers rdma=$rdma batch_size=$batch_size ./cifar10.sh

exclude_parts=CommunicateInverse
epochs=$epochs base_lr=$base_lr kfac=1 exclude_parts=$exclude_parts damping=$damping stat_decay=$stat_decay kfac_type=$kfac_type kfac_name=$kfac_name dnn=$dnn nworkers=$nworkers rdma=$rdma batch_size=$batch_size ./cifar10.sh

exclude_parts=CommunicateInverse,ComputeInverse
epochs=$epochs base_lr=$base_lr kfac=1 exclude_parts=$exclude_parts damping=$damping stat_decay=$stat_decay kfac_type=$kfac_type kfac_name=$kfac_name dnn=$dnn nworkers=$nworkers rdma=$rdma batch_size=$batch_size ./cifar10.sh

exclude_parts=CommunicateInverse,ComputeInverse,CommunicateFactor
epochs=$epochs base_lr=$base_lr kfac=1 exclude_parts=$exclude_parts damping=$damping stat_decay=$stat_decay kfac_type=$kfac_type kfac_name=$kfac_name dnn=$dnn nworkers=$nworkers rdma=$rdma batch_size=$batch_size ./cifar10.sh

exclude_parts=CommunicateInverse,ComputeInverse,CommunicateFactor,ComputeFactor
epochs=$epochs base_lr=$base_lr kfac=1 exclude_parts=$exclude_parts damping=$damping stat_decay=$stat_decay kfac_type=$kfac_type kfac_name=$kfac_name dnn=$dnn nworkers=$nworkers rdma=$rdma batch_size=$batch_size ./cifar10.sh


# ds-kfac
kfac_name=inverse_nordc

exclude_parts=''
epochs=$epochs base_lr=$base_lr kfac=1 exclude_parts=$exclude_parts damping=$damping stat_decay=$stat_decay kfac_type=$kfac_type kfac_name=$kfac_name dnn=$dnn nworkers=$nworkers rdma=$rdma batch_size=$batch_size ./cifar10.sh

exclude_parts=CommunicateInverse
epochs=$epochs base_lr=$base_lr kfac=1 exclude_parts=$exclude_parts damping=$damping stat_decay=$stat_decay kfac_type=$kfac_type kfac_name=$kfac_name dnn=$dnn nworkers=$nworkers rdma=$rdma batch_size=$batch_size ./cifar10.sh

exclude_parts=CommunicateInverse,ComputeInverse
epochs=$epochs base_lr=$base_lr kfac=1 exclude_parts=$exclude_parts damping=$damping stat_decay=$stat_decay kfac_type=$kfac_type kfac_name=$kfac_name dnn=$dnn nworkers=$nworkers rdma=$rdma batch_size=$batch_size ./cifar10.sh

exclude_parts=CommunicateInverse,ComputeInverse,CommunicateFactor,ComputeFactor
epochs=$epochs base_lr=$base_lr kfac=1 exclude_parts=$exclude_parts damping=$damping stat_decay=$stat_decay kfac_type=$kfac_type kfac_name=$kfac_name dnn=$dnn nworkers=$nworkers rdma=$rdma batch_size=$batch_size ./cifar10.sh

