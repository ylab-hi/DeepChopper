#!/bin/bash
set -e
set -u
set -o pipefail

# --dataset_name tests/data/test_input.parquet
dc-hg-train \
	--hyenadna_model hyenadna-small-32k-seqlen \
	--train_file data/train.parquet \
	--validation_file data/val.parquet \
	--test_file data/test.parquet \
	--output_dir notebooks/hyena_model_train \
	--num_train_epochs 1 \
	--learning_rate 2e-5 \
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
	--run_name deepchopper_train \
	--do_train --do_eval --do_predict
