horovod=0
nworkers=8
rdma=1
node_rank=1
node_count=2

dnn=resnet50
batch_size=32
epochs=1

kfac_name=inverse
kfac=1
fac=1

# s-sgd
kfac=0 horovod=$horovod nworkers=$nworkers rdma=$rdma node_rank=$node_rank node_count=$node_count batch_size=$batch_size epochs=$epochs ./train_imagenet.sh

# mpd-kfac
kfac=$kfac fac=$fac kfac_name=$kfac_name horovod=$horovod nworkers=$nworkers rdma=$rdma node_rank=$node_rank node_count=$node_count batch_size=$batch_size epochs=$epochs ./train_imagenet.sh

