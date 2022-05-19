# The effects of damping and the update frequency of KFs on ResNet-110 with eigen and eigen-dp

#kfac_name=eigen_dp
kfac_name=eigen

#horovod=0 node_rank=1 kfac_name=$kfac_name fac=1 kfac=1 damping=0.005 bash train_cifar10.sh &
#horovod=0 node_rank=2 kfac_name=$kfac_name fac=1 kfac=1 damping=0.01 bash train_cifar10.sh &
#horovod=0 node_rank=3 kfac_name=$kfac_name fac=1 kfac=1 damping=0.05 bash train_cifar10.sh &
#horovod=0 node_rank=4 kfac_name=$kfac_name fac=1 kfac=1 damping=0.1 bash train_cifar10.sh &

#horovod=0 node_rank=5 kfac_name=$kfac_name fac=10 kfac=10 damping=0.005 bash train_cifar10.sh &
#horovod=0 node_rank=6 kfac_name=$kfac_name fac=10 kfac=10 damping=0.01 bash train_cifar10.sh &
#horovod=0 node_rank=7 kfac_name=$kfac_name fac=10 kfac=10 damping=0.05 bash train_cifar10.sh &
#horovod=0 node_rank=8 kfac_name=$kfac_name fac=10 kfac=10 damping=0.1 bash train_cifar10.sh &

#horovod=0 node_rank=9 kfac_name=$kfac_name fac=50 kfac=50 damping=0.005 bash train_cifar10.sh &
#horovod=0 node_rank=10 kfac_name=$kfac_name fac=50 kfac=50 damping=0.01 bash train_cifar10.sh &
#horovod=0 node_rank=11 kfac_name=$kfac_name fac=50 kfac=50 damping=0.05 bash train_cifar10.sh &
#horovod=0 node_rank=12 kfac_name=$kfac_name fac=50 kfac=50 damping=0.1 bash train_cifar10.sh &

#horovod=0 node_rank=13 kfac_name=$kfac_name fac=100 kfac=100 damping=0.005 bash train_cifar10.sh &
#horovod=0 node_rank=14 kfac_name=$kfac_name fac=100 kfac=100 damping=0.01 bash train_cifar10.sh &
#horovod=0 node_rank=15 kfac_name=$kfac_name fac=100 kfac=100 damping=0.05 bash train_cifar10.sh &
#horovod=0 node_rank=16 kfac_name=$kfac_name fac=100 kfac=100 damping=0.1 bash train_cifar10.sh &

