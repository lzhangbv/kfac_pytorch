# Example:
# gpu=0 lr_mul=2 scale_emb_or_prj=prj bash train_multi30k_de_en_single.sh
# The better setting is:
# gpu=0 lr_mul=0.5 scale_emb_or_prj=emb bash train_multi30k_de_en_single.sh

CUDA_VISIBLE_DEVICES=${gpu} python examples/pytorch_multi30k_de_en_transformer_single.py \
-data_pkl data/m30k_deen_shr.pkl \
-label_smoothing \
-proj_share_weight \
-scale_emb_or_prj ${scale_emb_or_prj} \
-lr_mul ${lr_mul} \
-b 256 \
-warmup 4000 \
-epoch 200 \
-seed 1 \
-output_dir output/lr_mul_${lr_mul}-scale_${scale_emb_or_prj} \
-use_tb
