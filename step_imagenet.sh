kfac_name=inverse_rdc

dnn=resnet50
batch_size=8
nworkers=16
epochs=1 kfac_name=$kfac_name kfac=1 dnn=$dnn nworkers=$nworkers rdma=1 batch_size=$batch_size ./imagenet.sh
