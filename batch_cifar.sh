dnn=resnet56
batch_size=50
nworkers=4
rdma=1
base_lr=0.1
epochs=1 #100

kfac_type=Femp
stat_decay=0.5
damping=0.00005
#exclude_parts=CommunicateInverse,ComputeInverse,CommunicateFactor,ComputeFactor
exclude_parts=''

# s-sgd
#epochs=$epochs base_lr=$base_lr kfac=0 dnn=$dnn nworkers=$nworkers rdma=$rdma batch_size=$batch_size ./train_cifar10.sh

# mpd-kfac-inv
kfac_name=inverse
epochs=$epochs base_lr=$base_lr kfac=1 exclude_parts=$exclude_parts kfac_type=$kfac_type kfac_name=$kfac_name dnn=$dnn nworkers=$nworkers rdma=$rdma batch_size=$batch_size ./train_cifar10.sh

# mpd-kfac-eigen
kfac_name=eigen
#epochs=$epochs base_lr=$base_lr kfac=1 exclude_parts=$exclude_parts kfac_type=$kfac_type kfac_name=$kfac_name dnn=$dnn nworkers=$nworkers rdma=$rdma batch_size=$batch_size ./train_cifar10.sh

# dp-kfac-inv
kfac_name=inverse_dp
#epochs=$epochs base_lr=$base_lr kfac=1 exclude_parts=$exclude_parts kfac_type=$kfac_type kfac_name=$kfac_name dnn=$dnn nworkers=$nworkers rdma=$rdma batch_size=$batch_size ./train_cifar10.sh

# dp-kfac-eigen
kfac_name=eigen_dp
#epochs=$epochs base_lr=$base_lr kfac=1 exclude_parts=$exclude_parts kfac_type=$kfac_type kfac_name=$kfac_name dnn=$dnn nworkers=$nworkers rdma=$rdma batch_size=$batch_size ./train_cifar10.sh

# dp-kfac-inv block
kfac_name=inverse_dp_block
#epochs=$epochs base_lr=$base_lr kfac=1 exclude_parts=$exclude_parts kfac_type=$kfac_type kfac_name=$kfac_name dnn=$dnn nworkers=$nworkers rdma=$rdma batch_size=$batch_size ./train_cifar10.sh
