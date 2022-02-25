horovod=0
nworkers=4
rdma=1
node_rank=1
node_count=1

dnn=resnet110
batch_size=128
epochs=100
base_lr=0.1
damping=0.003

# s-sgd
#kfac=0 horovod=$horovod nworkers=$nworkers rdma=$rdma node_rank=$node_rank node_count=$node_count dnn=$dnn batch_size=$batch_size epochs=$epochs base_lr=$base_lr damping=$damping bash train_cifar10.sh

fac=1
kfac=1

kfac_name=inverse
#kfac=$kfac fac=$fac kfac_name=$kfac_name horovod=$horovod nworkers=$nworkers rdma=$rdma node_rank=$node_rank node_count=$node_count dnn=$dnn batch_size=$batch_size epochs=$epochs base_lr=$base_lr damping=$damping bash train_cifar10.sh

kfac_name=eigen
#kfac=$kfac fac=$fac kfac_name=$kfac_name horovod=$horovod nworkers=$nworkers rdma=$rdma node_rank=$node_rank node_count=$node_count dnn=$dnn batch_size=$batch_size epochs=$epochs base_lr=$base_lr damping=$damping bash train_cifar10.sh

kfac_name=inverse_dp
#kfac=$kfac fac=$fac kfac_name=$kfac_name horovod=$horovod nworkers=$nworkers rdma=$rdma node_rank=$node_rank node_count=$node_count dnn=$dnn batch_size=$batch_size epochs=$epochs base_lr=$base_lr damping=$damping bash train_cifar10.sh

kfac_name=eigen_dp
kfac=$kfac fac=$fac kfac_name=$kfac_name horovod=$horovod nworkers=$nworkers rdma=$rdma node_rank=$node_rank node_count=$node_count dnn=$dnn batch_size=$batch_size epochs=$epochs base_lr=$base_lr damping=$damping bash train_cifar10.sh

kfac_name=inverse_dp_block
#kfac=$kfac fac=$fac kfac_name=$kfac_name horovod=$horovod nworkers=$nworkers rdma=$rdma node_rank=$node_rank node_count=$node_count dnn=$dnn batch_size=$batch_size epochs=$epochs base_lr=$base_lr damping=$damping bash train_cifar10.sh

kfac_name=inverse_kaisa
#kfac=$kfac fac=$fac kfac_name=$kfac_name horovod=$horovod nworkers=$nworkers rdma=$rdma node_rank=$node_rank node_count=$node_count dnn=$dnn batch_size=$batch_size epochs=$epochs base_lr=$base_lr damping=$damping bash train_cifar10.sh


