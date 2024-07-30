#!/bin/bash
set -euo pipefail # Combines the set commands into one line

# Paths to checkpoint files
hyena_ckpt_path="/projects/b1171/ylk4626/project/DeepChopper/logs/train/runs/2024-04-08_23-19-20/checkpoints/epoch_005_f1_0.9933.ckpt"

# rna 004 only
# hyena_ckpt_path="/projects/b1171/ylk4626/project/DeepChopper/logs/train/runs/2024-07-02_15-20-53/checkpoints/epoch_008_f1_0.9946.ckpt"

cnn_ckpt_path="/projects/b1171/ylk4626/project/DeepChopper/logs/train/runs/2024-04-07_12-01-37/checkpoints/epoch_036_f1_0.9914.ckpt"
caduceus_ckpt_path="/projects/b1171/ylk4626/project/DeepChopper/logs/train/runs/2024-05-25_19-42-45/checkpoints/epoch_002_f1_0.9982.ckpt"

# Default model selection
sample_name="MCF7_hyena_model"
data_folder="data/dorado_without_trim_fqs/MCF7.fastq_chunks"

num_workers=60
batch_size=64

# get the model from the command line

if [ "$#" -eq 0 ]; then
	echo "Please select a valid model: cnn, hyena, or caduceus."
	exit 1
elif [ "$#" -eq 1 ]; then
	model=$1
else
	echo "Error: Too many arguments."
	echo "Please provide only one argument: cnn, hyena, or caduceus."
	exit 1
fi

# Set the checkpoint path based on the selected model
if [ "$model" = "cnn" ]; then
	ckpt_path="$cnn_ckpt_path"
elif [ "$model" = "hyena" ]; then
	ckpt_path="$hyena_ckpt_path"
elif [ "$model" = "caduceus" ]; then
	ckpt_path="$caduceus_ckpt_path"
else
	echo "Error: Model not supported."
	echo "Please select a valid model: cnn, hyena, or caduceus."
	exit 1
fi

# Iterate over each .parquet file in the data folder and evaluate
for data_path in "$data_folder"/*.parquet; do
	# Extract filename without extension
	filename=$(basename -- "$data_path")
	filename="${filename%.*}"

	# Define output directory using the filename
	output_dir="/projects/b1171/ylk4626/project/DeepChopper/logs/eval/runs/${sample_name}_${model}/${filename}"

	echo "Evaluating file: $data_path"
	echo "Output directory: $output_dir"

	poe eval \
		ckpt_path="$ckpt_path" \
		model="$model" \
		+data.predict_data_path="$data_path" \
		trainer=gpu \
		data.num_workers=$num_workers \
		data.batch_size=$batch_size \
		paths.output_dir="$output_dir" \
		paths.log_dir="$output_dir" \
		tags=["eval"] \
		extras.print_config=False
done
