export CUDA_VISIBLE_DEVICES=4

nohup python train_partial_data.py --session_name TPAVI_S4_$(basename $0) --rest_frac 0.5 --visual_backbone resnet --train_batch_size 4 --lr 0.0001 \
--tpavi_stages 0 1 2 3 --tpavi_va_flag --masked_av_flag --masked_av_stages 0 1 2 3 --lambda_1 0.5 --kl_flag --max_epoches 50 > /dev/null 2>&1 & 