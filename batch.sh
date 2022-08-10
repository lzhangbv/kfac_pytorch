# Shell script to run experiments of convergence performance and training efficiency. 
# Note:
#     (1) Before run convergence experiments, make sure SPEED=False in example scripts;
#         or make sure SPEED=True in example scripts before run efficiency experiments. 
#     (2) Please fine-tune the hyper-paramters in train_xxx.sh scripts. 
#     (3) Please configure the host files and the cluster environments in configs folder. 


# Convergence performance
#bash train_cifar10.sh
#bash train_cifar100.sh
#bash train_imagenet.sh
#bash train_multi30k.sh
#bash train_squad.sh

# Training efficiency
# SGD/Adam
kfac=0
use_adam=1

# K-FAC
fac=1
kfac=1
kfac_name=eigen_dp #eigen_dp # choices: inverse, inverse_dp, eigen, eigen_dp

#kfac=$kfac fac=$fac kfac_name=$kfac_name epochs=1 dnn=resnet110 batch_size=128 nworkers=4 bash train_cifar10.sh
#kfac=$kfac fac=$fac kfac_name=$kfac_name epochs=1 dnn=vgg16 batch_size=128 nworkers=4 bash train_cifar100.sh
#kfac=$kfac fac=$fac kfac_name=$kfac_name epochs=1 dnn=resnet50 batch_size=32 nworkers=64 bash train_imagenet.sh
#kfac=$kfac fac=$fac kfac_name=$kfac_name epochs=1 dnn=densenet201 batch_size=16 nworkers=64 bash train_imagenet.sh
#kfac=$kfac fac=$fac kfac_name=$kfac_name epochs=1 dnn=inceptionv4 batch_size=16 nworkers=64 bash train_imagenet.sh
#kfac=$kfac fac=$fac kfac_name=$kfac_name use_adam=$use_adam epochs=5 n_layers=6 batch_size=128 nworkers=8 bash train_multi30k.sh
#kfac=$kfac fac=$fac kfac_name=$kfac_name use_adamw=$use_adam epochs=1 batch_size=4 nworkers=8 bash train_squad.sh
