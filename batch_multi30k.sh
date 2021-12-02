horovod=0
nworkers=8
rdma=1
node_rank=1
node_count=2

n_layers=2 # shallow=2, standard=6
batch_size=128

# sgd
epochs=200
base_lr=1e-6
warmup_epochs=0
lr_decay="${lr_decay:-200}"

#kfac=0 use_adam=0 horovod=$horovod nworkers=$nworkers rdma=$rdma node_rank=$node_rank node_count=$node_count n_layers=$n_layers base_lr=$base_lr warmup_epochs=$warmup_epochs lr_decay=$lr_decay batch_size=$batch_size epochs=$epochs bash train_multi30k.sh

# adam
epochs=200
lr_mul=0.5
warmup=4000

#kfac=0 use_adam=1 horovod=$horovod nworkers=$nworkers rdma=$rdma node_rank=$node_rank node_count=$node_count n_layers=$n_layers lr_mul=$lr_mul warmup=$warmup batch_size=$batch_size epochs=$epochs bash train_multi30k.sh

# kfac
epochs=50
base_lr=1e-6
warmup_epochs=0
lr_decay="${lr_decay:-200}"
kfac=10
fac=10

#kfac_name=inverse
#damping=0.01
kfac_name=inverse_dp
damping=0.001

kfac=$kfac fac=$fac use_adam=0 horovod=$horovod nworkers=$nworkers rdma=$rdma node_rank=$node_rank node_count=$node_count n_layers=$n_layers damping=$damping kfac_name=$kfac_name base_lr=$base_lr warmup_epochs=$warmup_epochs lr_decay=$lr_decay batch_size=$batch_size epochs=$epochs bash train_multi30k.sh

