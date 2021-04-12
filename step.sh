kfac_type=Femp

#exclude_parts=CommunicateFactor
exclude_parts=''

dnn=resnet56
batch_size=128
nworkers=4
rdma=0
base_lr=0.1
epochs=100

#epochs=$epochs base_lr=$base_lr kfac=0 dnn=$dnn nworkers=$nworkers rdma=$rdma batch_size=$batch_size ./cifar10.sh

stat_decay=0.5
damping=0.00005

kfac_name=inverse_nordc
#epochs=$epochs base_lr=$base_lr kfac=1 exclude_parts=$exclude_parts damping=$damping stat_decay=$stat_decay kfac_type=$kfac_type kfac_name=$kfac_name dnn=$dnn nworkers=$nworkers rdma=$rdma batch_size=$batch_size ./cifar10.sh

kfac_name=inverse
#epochs=$epochs base_lr=$base_lr kfac=1 exclude_parts=$exclude_parts damping=$damping stat_decay=$stat_decay kfac_type=$kfac_type kfac_name=$kfac_name dnn=$dnn nworkers=$nworkers rdma=$rdma batch_size=$batch_size ./cifar10.sh


nworkers=1
epochs=1

batch_size=8
epochs=$epochs base_lr=$base_lr kfac=0 dnn=$dnn nworkers=$nworkers rdma=$rdma batch_size=$batch_size ./cifar10.sh

batch_size=16
epochs=$epochs base_lr=$base_lr kfac=0 dnn=$dnn nworkers=$nworkers rdma=$rdma batch_size=$batch_size ./cifar10.sh

batch_size=32
epochs=$epochs base_lr=$base_lr kfac=0 dnn=$dnn nworkers=$nworkers rdma=$rdma batch_size=$batch_size ./cifar10.sh

batch_size=64
epochs=$epochs base_lr=$base_lr kfac=0 dnn=$dnn nworkers=$nworkers rdma=$rdma batch_size=$batch_size ./cifar10.sh

batch_size=128
epochs=$epochs base_lr=$base_lr kfac=0 dnn=$dnn nworkers=$nworkers rdma=$rdma batch_size=$batch_size ./cifar10.sh

batch_size=256
epochs=$epochs base_lr=$base_lr kfac=0 dnn=$dnn nworkers=$nworkers rdma=$rdma batch_size=$batch_size ./cifar10.sh

batch_size=512
epochs=$epochs base_lr=$base_lr kfac=0 dnn=$dnn nworkers=$nworkers rdma=$rdma batch_size=$batch_size ./cifar10.sh

batch_size=1024
epochs=$epochs base_lr=$base_lr kfac=0 dnn=$dnn nworkers=$nworkers rdma=$rdma batch_size=$batch_size ./cifar10.sh

