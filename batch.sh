# Shell script to run experiments of convergence performance and training efficiency. 
# Note:
#     (1) Before run convergence experiments, make sure SPEED=False in example scripts;
#         or make sure SPEED=True in example scripts before run efficiency experiments. 
#     (2) Please fine-tune the hyper-paramters in train_xxx.sh scripts. 
#     (3) Please configure the host files and the cluster environments in configs folder. 

fac=1
kfac=1
kfac_name=fast

# Convergence performance
#kfac=$kfac fac=$fac kfac_name=$kfac_name bash train_cifar10.sh
#bash train_cifar100.sh
#bash train_imagenet.sh
#bash train_multi30k.sh
#bash train_squad.sh


# Training efficiency
#kfac=$kfac fac=$fac kfac_name=$kfac_name epochs=1 dnn=resnet110 batch_size=128 nworkers=4 bash train_cifar10.sh
#kfac=$kfac fac=$fac kfac_name=$kfac_name epochs=1 dnn=vgg16 batch_size=128 nworkers=4 bash train_cifar100.sh
#kfac=$kfac fac=$fac kfac_name=$kfac_name epochs=1 dnn=resnet50 batch_size=32 nworkers=64 bash train_imagenet.sh
#kfac=$kfac fac=$fac kfac_name=$kfac_name epochs=1 dnn=densenet201 batch_size=16 nworkers=64 bash train_imagenet.sh
#kfac=$kfac fac=$fac kfac_name=$kfac_name epochs=1 dnn=inceptionv4 batch_size=16 nworkers=64 bash train_imagenet.sh
#kfac=$kfac fac=$fac kfac_name=$kfac_name epochs=5 n_layers=6 batch_size=128 nworkers=8 bash train_multi30k.sh
#kfac=$kfac fac=$fac kfac_name=$kfac_name epochs=1 batch_size=4 nworkers=8 bash train_squad.sh


# Tuning the hyper-parameters

epochs=100 dnn=resnet110 dataset=cifar10 batch_size=128 lr=0.1 nworkers=4 kfac_name=fast kfac=1 fac=1 damping=0.003 stat_decay=0.05 kl_clip=0.001 warmup_epochs=1 clusterprefix=gpu1cluster ./train_cifar10.sh &
epochs=100 dnn=resnet110 dataset=cifar10 batch_size=128 lr=0.1 nworkers=4 kfac_name=fast kfac=1 fac=1 damping=0.003 stat_decay=0.05 kl_clip=0.001 warmup_epochs=5 clusterprefix=gpu2cluster ./train_cifar10.sh &
epochs=100 dnn=resnet110 dataset=cifar10 batch_size=128 lr=0.1 nworkers=4 kfac_name=fast kfac=1 fac=1 damping=0.003 stat_decay=0.05 kl_clip=0.005 warmup_epochs=1 clusterprefix=gpu3cluster ./train_cifar10.sh &
epochs=100 dnn=resnet110 dataset=cifar10 batch_size=128 lr=0.1 nworkers=4 kfac_name=fast kfac=1 fac=1 damping=0.003 stat_decay=0.05 kl_clip=0.005 warmup_epochs=5 clusterprefix=gpu4cluster ./train_cifar10.sh &
epochs=100 dnn=resnet110 dataset=cifar10 batch_size=128 lr=0.1 nworkers=4 kfac_name=fast kfac=1 fac=1 damping=0.003 stat_decay=0.95 kl_clip=0.001 warmup_epochs=1 clusterprefix=gpu5cluster ./train_cifar10.sh &
epochs=100 dnn=resnet110 dataset=cifar10 batch_size=128 lr=0.1 nworkers=4 kfac_name=fast kfac=1 fac=1 damping=0.003 stat_decay=0.95 kl_clip=0.001 warmup_epochs=5 clusterprefix=gpu6cluster ./train_cifar10.sh &
epochs=100 dnn=resnet110 dataset=cifar10 batch_size=128 lr=0.1 nworkers=4 kfac_name=fast kfac=1 fac=1 damping=0.003 stat_decay=0.95 kl_clip=0.005 warmup_epochs=1 clusterprefix=gpu7cluster ./train_cifar10.sh &
epochs=100 dnn=resnet110 dataset=cifar10 batch_size=128 lr=0.1 nworkers=4 kfac_name=fast kfac=1 fac=1 damping=0.003 stat_decay=0.95 kl_clip=0.005 warmup_epochs=5 clusterprefix=gpu8cluster ./train_cifar10.sh &

epochs=100 dnn=resnet110 dataset=cifar10 batch_size=128 lr=0.1 nworkers=4 kfac_name=fast kfac=1 fac=1 damping=0.03 stat_decay=0.05 kl_clip=0.001 warmup_epochs=1 clusterprefix=gpu9cluster ./train_cifar10.sh &
epochs=100 dnn=resnet110 dataset=cifar10 batch_size=128 lr=0.1 nworkers=4 kfac_name=fast kfac=1 fac=1 damping=0.03 stat_decay=0.05 kl_clip=0.001 warmup_epochs=5 clusterprefix=gpu10cluster ./train_cifar10.sh &
epochs=100 dnn=resnet110 dataset=cifar10 batch_size=128 lr=0.1 nworkers=4 kfac_name=fast kfac=1 fac=1 damping=0.03 stat_decay=0.05 kl_clip=0.005 warmup_epochs=1 clusterprefix=gpu11cluster ./train_cifar10.sh &
epochs=100 dnn=resnet110 dataset=cifar10 batch_size=128 lr=0.1 nworkers=4 kfac_name=fast kfac=1 fac=1 damping=0.03 stat_decay=0.05 kl_clip=0.005 warmup_epochs=5 clusterprefix=gpu12cluster ./train_cifar10.sh &
epochs=100 dnn=resnet110 dataset=cifar10 batch_size=128 lr=0.1 nworkers=4 kfac_name=fast kfac=1 fac=1 damping=0.03 stat_decay=0.95 kl_clip=0.001 warmup_epochs=1 clusterprefix=gpu13cluster ./train_cifar10.sh &
epochs=100 dnn=resnet110 dataset=cifar10 batch_size=128 lr=0.1 nworkers=4 kfac_name=fast kfac=1 fac=1 damping=0.03 stat_decay=0.95 kl_clip=0.001 warmup_epochs=5 clusterprefix=gpu14cluster ./train_cifar10.sh &
epochs=100 dnn=resnet110 dataset=cifar10 batch_size=128 lr=0.1 nworkers=4 kfac_name=fast kfac=1 fac=1 damping=0.03 stat_decay=0.95 kl_clip=0.005 warmup_epochs=1 clusterprefix=gpu15cluster ./train_cifar10.sh &
epochs=100 dnn=resnet110 dataset=cifar10 batch_size=128 lr=0.1 nworkers=4 kfac_name=fast kfac=1 fac=1 damping=0.03 stat_decay=0.95 kl_clip=0.005 warmup_epochs=5 clusterprefix=gpu16cluster ./train_cifar10.sh &

