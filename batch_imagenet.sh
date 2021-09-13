#kfac_name=eigen
kfac_name=inverse

dnn=resnet50
batch_size=32
nworkers=64

# sgd
#epochs=1 kfac_name=$kfac_name kfac=0 dnn=$dnn nworkers=1 rdma=1 batch_size=$batch_size ./train_imagenet.sh

# s-sgd
#epochs=1 kfac_name=$kfac_name kfac=0 dnn=$dnn nworkers=$nworkers rdma=1 batch_size=$batch_size ./train_imagenet.sh

# kfac
#epochs=1 kfac_name=$kfac_name kfac=1 dnn=$dnn nworkers=1 rdma=1 batch_size=$batch_size ./train_imagenet.sh

exclude_parts=CommunicateInverse,ComputeInverse
#epochs=1 exclude_parts=$exclude_parts kfac_name=$kfac_name kfac=1 dnn=$dnn nworkers=1 rdma=1 batch_size=$batch_size ./train_imagenet.sh

# mpd-kfac
epochs=1 kfac_name=$kfac_name kfac=1 dnn=$dnn nworkers=$nworkers rdma=1 batch_size=$batch_size ./train_imagenet.sh

exclude_parts=CommunicateInverse
#epochs=1 exclude_parts=$exclude_parts kfac_name=$kfac_name kfac=1 dnn=$dnn nworkers=$nworkers rdma=1 batch_size=$batch_size ./train_imagenet.sh

exclude_parts=CommunicateInverse,ComputeInverse
#epochs=1 exclude_parts=$exclude_parts kfac_name=$kfac_name kfac=1 dnn=$dnn nworkers=$nworkers rdma=1 batch_size=$batch_size ./train_imagenet.sh

exclude_parts=CommunicateInverse,ComputeInverse,CommunicateFactor
#epochs=1 exclude_parts=$exclude_parts kfac_name=$kfac_name kfac=1 dnn=$dnn nworkers=$nworkers rdma=1 batch_size=$batch_size ./train_imagenet.sh

exclude_parts=CommunicateInverse,ComputeInverse,CommunicateFactor,ComputeFactor
#epochs=1 exclude_parts=$exclude_parts kfac_name=$kfac_name kfac=1 dnn=$dnn nworkers=$nworkers rdma=1 batch_size=$batch_size ./train_imagenet.sh
