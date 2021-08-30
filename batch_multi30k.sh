nworkers=16
batch_size=128
lr_mul=0.5
warmup=500
scale_emb_or_prj=emb

# adam
epochs=100
nworkers=$nworkers kfac=0 lr_mul=$lr_mul warmup=$warmup scale_emb_or_prj=$scale_emb_or_prj batch_size=$batch_size epochs=$epochs bash train_multi30k.sh

# kfac
kfac_name=eigen
epochs=100
damping=0.003

nworkers=$nworkers kfac=1 kfac_name=$kfac_name lr_mul=$lr_mul warmup=$warmup scale_emb_or_prj=$scale_emb_or_prj batch_size=$batch_size epochs=$epochs bash train_multi30k.sh
