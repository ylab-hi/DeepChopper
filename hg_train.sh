#!/bin/bash
set -e
set -u
set -o pipefail

# --dataset_name tests/data/test_input.parquet
# --max_train_sample 1000 \
# --max_eval_samples 100 \
# --max_predict_samples 100 \

# 24 batch size for 60GB GPU

inputdir="data/60_0000_samples"
# outdirname="cdc_train100000_20ep_18b"
outdirname="sg_data_train_600000_20ep_8b"
# --resume_from_checkpoint notebooks/$outdirname/checkpoint-213336 \

# accelerate launch
python hg_train.py \
	--hyenadna_model hyenadna-small-32k-seqlen \
	--train_file $inputdir/train.parquet \
	--validation_file $inputdir/val.parquet \
	--test_file $inputdir/test.parquet \
	--max_eval_sample 2000 \
	--max_predict_samples 2000 \
	--output_dir notebooks/$outdirname \
	--resume_from_checkpoint notebooks/$outdirname/checkpoint-30000 \
	--num_train_epochs 20 \
	--learning_rate 2e-5 \
	--gradient_accumulation_steps 1 \
	--per_device_train_batch_size 8 \
	--per_device_eval_batch_size 8 \
	--weight_decay 0.01 \
	--torch_compile False \
	--push_to_hub False \
	--evaluation_strategy epoch \
	--save_strategy epoch \
	--load_best_model_at_end \
	--trust_remote_code \
	--report_to wandb \
	--run_name $outdirname \
	--do_train --do_eval --do_predict
