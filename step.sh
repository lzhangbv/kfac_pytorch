kfac_type=Femp
kfac_name=inverse_nordc

#exclude_parts=CommunicateFactor
exclude_parts=''

dnn=resnet32
batch_size=128

#nworkers=4
#epochs=200 kfac=0 dnn=$dnn nworkers=$nworkers rdma=1 batch_size=$batch_size ./cifar10.sh

nworkers=4
epochs=100 kfac=10 exclude_parts=$exclude_parts kfac_type=$kfac_type kfac_name=$kfac_name dnn=$dnn nworkers=$nworkers rdma=1 batch_size=$batch_size ./cifar10.sh

