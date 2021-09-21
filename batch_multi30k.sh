nworkers=8 #8
batch_size=128

n_layers=2 # shallow=2, standard=6

# sgd
epochs=200
use_adam=0
base_lr=1e-6
warmup_epochs=0
lr_decay="${lr_decay:-200}"
# nworkers=$nworkers n_layers=$n_layers use_adam=$use_adam kfac=0 base_lr=$base_lr warmup_epochs=$warmup_epochs lr_decay=$lr_decay batch_size=$batch_size epochs=$epochs bash train_multi30k.sh

# adam
epochs=200
use_adam=1
lr_mul=0.5
warmup=4000
# nworkers=$nworkers n_layers=$n_layers use_adam=$use_adam kfac=0 lr_mul=$lr_mul warmup=$warmup batch_size=$batch_size epochs=$epochs bash train_multi30k.sh

# kfac
epochs=200
use_adam=0
base_lr=1e-6
warmup_epochs=0
lr_decay="${lr_decay:-200}"
kfac_name=eigen
damping=0.001
kfac=5
fac=5
nworkers=$nworkers n_layers=$n_layers use_adam=$use_adam kfac=$kfac fac=$fac damping=$damping kfac_name=$kfac_name base_lr=$base_lr warmup_epochs=$warmup_epochs lr_decay=$lr_decay batch_size=$batch_size epochs=$epochs bash train_multi30k.sh

# use adam

