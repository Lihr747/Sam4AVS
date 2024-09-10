export CUDA_VISIBLE_DEVICES=2

nohup python train_partial_data.py --session_name MS3_$(basename $0) --visual_backbone audioclip_realfpn --rest_frac 0.2 --train_batch_size 4 --lr 0.00005 \
--freeze_audio_backbone --backbone_lr_mult 0.1 --max_epoches 200 --fusion concate --fpn_upmode bilinear > /dev/null 2>&1 &
# for fusion we use 1024 7 7 to replace 2048 7 7 > /dev/