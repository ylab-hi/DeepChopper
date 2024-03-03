#!/bin/bash
set -e
set -u
set -o pipefail

dc-hg-train --hyenadna_model hyenadna-small-32k-seqlen \
	--dataset_name tests/data/test_input.parquet \
	--learning_rate 2e-5 \
	--per_device_train_batch_size 8 \
	--per_device_eval_batch_size 8 \
	--weight_decay 0.01 \
	--torch_compile False \
	--push_to_hub False \
	--num_train_epochs 1 \
	--weight_decay 0.01 \
	--output_dir notebooks/hyena_model_train \
	--evaluation_strategy epoch \
	--save_strategy epoch \
	--load_best_model_at_end \
	--trust_remote_code \
	--report_to wandb \
	--run_name deepchopper \
	--do_train --do_eval --do_predict
