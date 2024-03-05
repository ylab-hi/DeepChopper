#!/bin/bash
set -e
set -u
set -o pipefail

# --dataset_name tests/data/test_input.parquet
# --max_train_sample 1000 \
# --max_eval_samples 100 \
# --max_predict_samples 100 \
# 24 batch size for 60GB GPU
dc-hg-train \
	--hyenadna_model hyenadna-small-32k-seqlen \
	--train_file data/600000_samples/train.parquet \
	--validation_file data/600000_samples/val.parquet \
	--test_file data/600000_samples/test.parquet \
	--output_dir notebooks/dc_train_600000_15_ep_18b \
	--num_train_epochs 15 \
	--learning_rate 2e-5 \
	--gradient_accumulation_steps 4 \
	--per_device_train_batch_size 18 \
	--per_device_eval_batch_size 18 \
	--weight_decay 0.01 \
	--torch_compile False \
	--push_to_hub False \
	--evaluation_strategy epoch \
	--save_strategy epoch \
	--load_best_model_at_end \
	--trust_remote_code \
	--report_to wandb \
	--run_name dc_train_600000_15_ep_18b \
	--do_train --do_eval --do_predict
