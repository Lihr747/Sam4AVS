export CUDA_VISIBLE_DEVICES=1

nohup python train_partial_data.py --session_name TPAVI_S4_$(basename $0) --rest_frac 0.2 --visual_backbone resnet --train_batch_size 4 --lr 0.0001 \
--tpavi_stages 0 1 2 3 --tpavi_va_flag --max_epoches 50 > /dev/null 2>&1 &