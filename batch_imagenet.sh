horovod=1
nworkers=64
rdma=1
node_rank=1
node_count=16


##### Iteration Time #####
dnn=resnet50
batch_size=32

#dnn=densenet201
#batch_size=16

#dnn=inceptionv4
#batch_size=16

epochs=1
kfac=1
fac=1

# sgd
#kfac=0 horovod=$horovod nworkers=1 rdma=$rdma node_rank=$node_rank node_count=$node_count dnn=$dnn batch_size=$batch_size epochs=$epochs ./train_imagenet.sh
#kfac=0 horovod=$horovod nworkers=$nworkers rdma=$rdma node_rank=$node_rank node_count=$node_count dnn=$dnn batch_size=$batch_size epochs=$epochs ./train_imagenet.sh

# kaisa
kfac_name=inverse_kaisa
#kfac=$kfac fac=$fac kfac_name=$kfac_name horovod=$horovod nworkers=$nworkers rdma=$rdma node_rank=$node_rank node_count=$node_count dnn=$dnn batch_size=$batch_size epochs=$epochs ./train_imagenet.sh
#exclude_parts=CommunicateInverse kfac=$kfac fac=$fac kfac_name=$kfac_name horovod=$horovod nworkers=$nworkers rdma=$rdma node_rank=$node_rank node_count=$node_count dnn=$dnn batch_size=$batch_size epochs=$epochs ./train_imagenet.sh
#exclude_parts=CommunicateInverse,ComputeInverse kfac=$kfac fac=$fac kfac_name=$kfac_name horovod=$horovod nworkers=$nworkers rdma=$rdma node_rank=$node_rank node_count=$node_count dnn=$dnn batch_size=$batch_size epochs=$epochs ./train_imagenet.sh
#exclude_parts=CommunicateInverse,ComputeInverse,CommunicateFactor kfac=$kfac fac=$fac kfac_name=$kfac_name horovod=$horovod nworkers=$nworkers rdma=$rdma node_rank=$node_rank node_count=$node_count dnn=$dnn batch_size=$batch_size epochs=$epochs ./train_imagenet.sh


# spd-kfac
kfac_name=inverse_spd
#kfac=$kfac fac=$fac kfac_name=$kfac_name horovod=$horovod nworkers=$nworkers rdma=$rdma node_rank=$node_rank node_count=$node_count dnn=$dnn batch_size=$batch_size epochs=$epochs ./train_imagenet.sh
#exclude_parts=CommunicateInverse kfac=$kfac fac=$fac kfac_name=$kfac_name horovod=$horovod nworkers=$nworkers rdma=$rdma node_rank=$node_rank node_count=$node_count dnn=$dnn batch_size=$batch_size epochs=$epochs ./train_imagenet.sh
#exclude_parts=CommunicateInverse,ComputeInverse kfac=$kfac fac=$fac kfac_name=$kfac_name horovod=$horovod nworkers=$nworkers rdma=$rdma node_rank=$node_rank node_count=$node_count dnn=$dnn batch_size=$batch_size epochs=$epochs ./train_imagenet.sh

# dp-kfac
#kfac_name=eigen_dp
kfac_name=inverse_dp
#kfac=$kfac fac=$fac kfac_name=$kfac_name horovod=$horovod nworkers=$nworkers rdma=$rdma node_rank=$node_rank node_count=$node_count dnn=$dnn batch_size=$batch_size epochs=$epochs ./train_imagenet.sh
#exclude_parts=CommunicateInverse kfac=$kfac fac=$fac kfac_name=$kfac_name horovod=$horovod nworkers=$nworkers rdma=$rdma node_rank=$node_rank node_count=$node_count dnn=$dnn batch_size=$batch_size epochs=$epochs ./train_imagenet.sh
#exclude_parts=CommunicateInverse,ComputeInverse kfac=$kfac fac=$fac kfac_name=$kfac_name horovod=$horovod nworkers=$nworkers rdma=$rdma node_rank=$node_rank node_count=$node_count dnn=$dnn batch_size=$batch_size epochs=$epochs ./train_imagenet.sh
