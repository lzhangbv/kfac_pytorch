#kfac_name=inverse_naive
#kfac_name=inverse_pipeline
#kfac_name=inverse_opt
exclude_parts=CommunicateFactor

dnn=resnet32
batch_size=128

#nworkers=4
#epochs=200 kfac=0  dnn=$dnn nworkers=$nworkers rdma=1 batch_size=$batch_size ./cifar10.sh

nworkers=4
epochs=100 kfac=10 exclude_parts=$exclude_parts dnn=$dnn nworkers=$nworkers rdma=1 batch_size=$batch_size ./cifar10.sh
# remmember to change the lr_decay in cifar10.sh


#dnn=inceptionv4
#batch_size=16
#nworkers=4
#epochs=1 kfac_name=$kfac_name kfac=1 dnn=$dnn nworkers=$nworkers rdma=1 batch_size=$batch_size ./imagenet.sh
