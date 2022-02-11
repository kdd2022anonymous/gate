#!/bin/bash
export CUDA_VISIBLE_DEVICES="0,1"

# Install necessary py packages
echo "start downloading pip packages"
pip install -r requirements.txt

# Read related configurations from .conf file
. shell/training_settings.conf

# Create necessary folders to save model checkpoints, log files, etc.
mkdir -p ${output_dir_log}
mkdir -p ${output_dir_model}
mkdir -p ${output_dir_event}

# Define data folder to save prepared datasets and  temp pickle files
output_dir_data=${op_data_Path}/output
mkdir -p ${output_dir_log}


echo 'Training started.' 
startTime=$(date +%Y%m%d-%H:%M:%S)
startTime_s=$(date +%s)

python3 run_bert.py \
	--train_file=${output_dir_data}/train_tree.csv \
	--test_file=${output_dir_data}/test_tree.csv \
	--tensorboard=${output_dir_event} \
	--pretrained_model=${pretrained_ckpt_path} \
	--output_dir_model=${output_dir_model} \
	--output_dir_data=${output_dir_data} \
	--output_dir_log=${output_dir_log} \
	--vocab_file=${vocab_file} \
	--config_file=${config_file} \
	--max_seq_length=${max_seq_length} \
	--train_batch_size=$((single_batch * gpu_num)) \
	--eval_batch_size=$((single_batch * gpu_num)) \
	--num_of_epoch=${num_of_epoch}

endTime=$(date +%Y%m%d_%H%M%S)
endTime_s=$(date +%s)
sumTime=$((endTime_s - startTime_s))
echo 'Training finished. ' "$startTime ---> $endTime" "Total:$sumTime seconds"  

