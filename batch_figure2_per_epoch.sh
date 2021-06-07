dnn=resnet56
batch_size=512
rdma=1
base_lr=0.1
epochs=100

kfac_type=Femp
stat_decay=0.5
damping=0.00005
exclude_parts=''

nworkers=32
# s-sgd
epochs=$epochs base_lr=$base_lr kfac=0 dnn=$dnn nworkers=$nworkers rdma=$rdma batch_size=$batch_size ./cifar10.sh

# d-kfac
kfac_name=inverse_nopar
epochs=$epochs base_lr=$base_lr kfac=1 exclude_parts=$exclude_parts damping=$damping stat_decay=$stat_decay kfac_type=$kfac_type kfac_name=$kfac_name dnn=$dnn nworkers=$nworkers rdma=$rdma batch_size=$batch_size ./cifar10.sh

# mpd-kfac
kfac_name=inverse
epochs=$epochs base_lr=$base_lr kfac=1 exclude_parts=$exclude_parts damping=$damping stat_decay=$stat_decay kfac_type=$kfac_type kfac_name=$kfac_name dnn=$dnn nworkers=$nworkers rdma=$rdma batch_size=$batch_size ./cifar10.sh

# ds-kfac
kfac_name=inverse_nordc
#epochs=$epochs base_lr=$base_lr kfac=1 exclude_parts=$exclude_parts damping=$damping stat_decay=$stat_decay kfac_type=$kfac_type kfac_name=$kfac_name dnn=$dnn nworkers=$nworkers rdma=$rdma batch_size=$batch_size ./cifar10.sh

nworkers=16
# s-sgd
epochs=$epochs base_lr=$base_lr kfac=0 dnn=$dnn nworkers=$nworkers rdma=$rdma batch_size=$batch_size ./cifar10.sh

# d-kfac
kfac_name=inverse_nopar
epochs=$epochs base_lr=$base_lr kfac=1 exclude_parts=$exclude_parts damping=$damping stat_decay=$stat_decay kfac_type=$kfac_type kfac_name=$kfac_name dnn=$dnn nworkers=$nworkers rdma=$rdma batch_size=$batch_size ./cifar10.sh

# mpd-kfac
kfac_name=inverse
epochs=$epochs base_lr=$base_lr kfac=1 exclude_parts=$exclude_parts damping=$damping stat_decay=$stat_decay kfac_type=$kfac_type kfac_name=$kfac_name dnn=$dnn nworkers=$nworkers rdma=$rdma batch_size=$batch_size ./cifar10.sh

# ds-kfac
kfac_name=inverse_nordc
#epochs=$epochs base_lr=$base_lr kfac=1 exclude_parts=$exclude_parts damping=$damping stat_decay=$stat_decay kfac_type=$kfac_type kfac_name=$kfac_name dnn=$dnn nworkers=$nworkers rdma=$rdma batch_size=$batch_size ./cifar10.sh

nworkers=8
# s-sgd
epochs=$epochs base_lr=$base_lr kfac=0 dnn=$dnn nworkers=$nworkers rdma=$rdma batch_size=$batch_size ./cifar10.sh

# d-kfac
kfac_name=inverse_nopar
epochs=$epochs base_lr=$base_lr kfac=1 exclude_parts=$exclude_parts damping=$damping stat_decay=$stat_decay kfac_type=$kfac_type kfac_name=$kfac_name dnn=$dnn nworkers=$nworkers rdma=$rdma batch_size=$batch_size ./cifar10.sh

# mpd-kfac
kfac_name=inverse
epochs=$epochs base_lr=$base_lr kfac=1 exclude_parts=$exclude_parts damping=$damping stat_decay=$stat_decay kfac_type=$kfac_type kfac_name=$kfac_name dnn=$dnn nworkers=$nworkers rdma=$rdma batch_size=$batch_size ./cifar10.sh

# ds-kfac
kfac_name=inverse_nordc
#epochs=$epochs base_lr=$base_lr kfac=1 exclude_parts=$exclude_parts damping=$damping stat_decay=$stat_decay kfac_type=$kfac_type kfac_name=$kfac_name dnn=$dnn nworkers=$nworkers rdma=$rdma batch_size=$batch_size ./cifar10.sh

nworkers=4
# s-sgd
epochs=$epochs base_lr=$base_lr kfac=0 dnn=$dnn nworkers=$nworkers rdma=$rdma batch_size=$batch_size ./cifar10.sh

# d-kfac
kfac_name=inverse_nopar
epochs=$epochs base_lr=$base_lr kfac=1 exclude_parts=$exclude_parts damping=$damping stat_decay=$stat_decay kfac_type=$kfac_type kfac_name=$kfac_name dnn=$dnn nworkers=$nworkers rdma=$rdma batch_size=$batch_size ./cifar10.sh

# mpd-kfac
kfac_name=inverse
epochs=$epochs base_lr=$base_lr kfac=1 exclude_parts=$exclude_parts damping=$damping stat_decay=$stat_decay kfac_type=$kfac_type kfac_name=$kfac_name dnn=$dnn nworkers=$nworkers rdma=$rdma batch_size=$batch_size ./cifar10.sh

# ds-kfac
kfac_name=inverse_nordc
#epochs=$epochs base_lr=$base_lr kfac=1 exclude_parts=$exclude_parts damping=$damping stat_decay=$stat_decay kfac_type=$kfac_type kfac_name=$kfac_name dnn=$dnn nworkers=$nworkers rdma=$rdma batch_size=$batch_size ./cifar10.sh
