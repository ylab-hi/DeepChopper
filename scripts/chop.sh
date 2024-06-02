#!/bin/bash
set -euo pipefail # Combines the set commands into one line

predict_folder="vcap_caduceus"

cargo run --bin chop -r -- \
	--pdt /projects/b1171/ylk4626/project/DeepChopper/logs/eval/runs/${predict_folder}/VCaP.fastq_0/predicts/0 \
	--pdt /projects/b1171/ylk4626/project/DeepChopper/logs/eval/runs/${predict_folder}/VCaP.fastq_1/predicts/0 \
	--pdt /projects/b1171/ylk4626/project/DeepChopper/logs/eval/runs/${predict_folder}/VCaP.fastq_2/predicts/0 \
	--pdt /projects/b1171/ylk4626/project/DeepChopper/logs/eval/runs/${predict_folder}/VCaP.fastq_3/predicts/0 \
	--pdt /projects/b1171/ylk4626/project/DeepChopper/logs/eval/runs/${predict_folder}/VCaP.fastq_4/predicts/0 \
	--pdt /projects/b1171/ylk4626/project/DeepChopper/logs/eval/runs/${predict_folder}/VCaP.fastq_5/predicts/0 \
	--pdt /projects/b1171/ylk4626/project/DeepChopper/logs/eval/runs/${predict_folder}/VCaP.fastq_6/predicts/0 \
	--pdt /projects/b1171/ylk4626/project/DeepChopper/logs/eval/runs/${predict_folder}/VCaP.fastq_7/predicts/0 \
	--pdt /projects/b1171/ylk4626/project/DeepChopper/logs/eval/runs/${predict_folder}/VCaP.fastq_8/predicts/0 \
	--pdt /projects/b1171/ylk4626/project/DeepChopper/logs/eval/runs/${predict_folder}/VCaP.fastq_9/predicts/0 \
	--fq /projects/b1171/ylk4626/project/DeepChopper/data/eval/real_data/dorado_without_trim_fqs/VCaP.fastq \
	-d -t 20 -o vcap_caduceus_all
