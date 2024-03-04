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
	--train_file data/train.parquet \
	--validation_file data/val.parquet \
	--test_file data/test.parquet \
	--output_dir notebooks/deepchopper_train \
	--num_train_epochs 150 \
	--learning_rate 2e-5 \
	--per_device_train_batch_size 24 \
	--per_device_eval_batch_size 24 \
	--weight_decay 0.01 \
	--torch_compile False \
	--push_to_hub False \
	--evaluation_strategy epoch \
	--save_strategy epoch \
	--load_best_model_at_end \
	--trust_remote_code \
	--report_to wandb \
	--run_name deepchopper_train_200 \
	--do_train --do_eval --do_predict
