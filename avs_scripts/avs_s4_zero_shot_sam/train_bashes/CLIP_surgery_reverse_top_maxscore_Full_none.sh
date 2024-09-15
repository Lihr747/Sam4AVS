filename=${0##*/}
filename=${filename%.*}
echo $filename

export CUDA_VISIBLE_DEVICES=2

base_folder=/home/yujr/workstation/Audio-Visual-Seg/avsbench_data/AVS_ZS_results/$filename
mkdir $base_folder

nohup python test_point_prompt.py --log_dir $base_folder --point_strategy top --backbone CLIP_Surgery --reverse --noise none --model_type Full > $base_folder/res.out &
