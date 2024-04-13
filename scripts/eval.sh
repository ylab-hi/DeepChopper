#!/bin/bash
set -euo pipefail # Combines the set commands into one line

# Paths to checkpoint files
hyena_ckpt_path="/projects/b1171/ylk4626/project/DeepChopper/logs/train/runs/2024-04-08_23-19-20/checkpoints/epoch_005_f1_0.9933.ckpt"
cnn_ckpt_path="/projects/b1171/ylk4626/project/DeepChopper/logs/train/runs/2024-04-07_12-01-37/checkpoints/epoch_036_f1_0.9914.ckpt"

# Default model selection
model="hyena"
data_path="data/eval/real_data/dorado_without_trim_fqs/K562.fastq_chunks/K562.fastq_2.parquet"
num_workers=60
batch_size=24

# Set the checkpoint path based on the selected model
if [ "$model" = "cnn" ]; then
	ckpt_path="$cnn_ckpt_path"
elif [ "$model" = "hyena" ]; then
	ckpt_path="$hyena_ckpt_path"
else
	echo "Error: Model not supported."
	exit 1
fi

# Evaluate the model
poe eval \
	ckpt_path="$ckpt_path" \
	model="$model" \
	+data.predict_data_path="$data_path" \
	trainer=gpu \
	data.num_workers=$num_workers \
	data.batch_size=$batch_size \
	tags=["eval", "hyena", "k562"]
