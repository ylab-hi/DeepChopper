#!/bin/bash
set -euo pipefail # Combines the set commands into one line

# predict_folder="vcap_caduceus"

cargo run --bin deepchopper-chop -r -- \
	--pdt /projects/b1171/ylk4626/project/DeepChopper/logs/eval/runs/vcap002_hyena_both_model_hyena/VCaP.fastq_0/predicts/0/ \
	--pdt /projects/b1171/ylk4626/project/DeepChopper/logs/eval/runs/vcap002_hyena_both_model_hyena/VCaP.fastq_1/predicts/0/ \
	--pdt /projects/b1171/ylk4626/project/DeepChopper/logs/eval/runs/vcap002_hyena_both_model_hyena/VCaP.fastq_2/predicts/0/ \
	--pdt /projects/b1171/ylk4626/project/DeepChopper/logs/eval/runs/vcap002_hyena_both_model_hyena/VCaP.fastq_3/predicts/0/ \
	--pdt /projects/b1171/ylk4626/project/DeepChopper/logs/eval/runs/vcap002_hyena_both_model_hyena/VCaP.fastq_4/predicts/0/ \
	--pdt /projects/b1171/ylk4626/project/DeepChopper/logs/eval/runs/vcap002_hyena_both_model_hyena/VCaP.fastq_5/predicts/0/ \
	--pdt /projects/b1171/ylk4626/project/DeepChopper/logs/eval/runs/vcap002_hyena_both_model_hyena/VCaP.fastq_6/predicts/0/ \
	--pdt /projects/b1171/ylk4626/project/DeepChopper/logs/eval/runs/vcap002_hyena_both_model_hyena/VCaP.fastq_7/predicts/0/ \
	--pdt /projects/b1171/ylk4626/project/DeepChopper/logs/eval/runs/vcap002_hyena_both_model_hyena/VCaP.fastq_8/predicts/0/ \
	--pdt /projects/b1171/ylk4626/project/DeepChopper/logs/eval/runs/vcap002_hyena_both_model_hyena/VCaP.fastq_9/predicts/0/ \
	--fq data/dorado_without_trim_fqs/VCaP.fastq -t 10 -o vcap002_hyena_both_model


cargo run --bin deepchopper-chop -r -- \
	--pdt /projects/b1171/ylk4626/project/DeepChopper/logs/eval/runs/vcap002_dorado_with_trim_hyena/vcap002_dorado_with_trim.fastq_0/predicts/0/ \
	--pdt /projects/b1171/ylk4626/project/DeepChopper/logs/eval/runs/vcap002_dorado_with_trim_hyena/vcap002_dorado_with_trim.fastq_1/predicts/0/ \
	--pdt /projects/b1171/ylk4626/project/DeepChopper/logs/eval/runs/vcap002_dorado_with_trim_hyena/vcap002_dorado_with_trim.fastq_2/predicts/0/ \
	--pdt /projects/b1171/ylk4626/project/DeepChopper/logs/eval/runs/vcap002_dorado_with_trim_hyena/vcap002_dorado_with_trim.fastq_3/predicts/0/ \
	--pdt /projects/b1171/ylk4626/project/DeepChopper/logs/eval/runs/vcap002_dorado_with_trim_hyena/vcap002_dorado_with_trim.fastq_4/predicts/0/ \
	--pdt /projects/b1171/ylk4626/project/DeepChopper/logs/eval/runs/vcap002_dorado_with_trim_hyena/vcap002_dorado_with_trim.fastq_5/predicts/0/ \
	--pdt /projects/b1171/ylk4626/project/DeepChopper/logs/eval/runs/vcap002_dorado_with_trim_hyena/vcap002_dorado_with_trim.fastq_6/predicts/0/ \
	--pdt /projects/b1171/ylk4626/project/DeepChopper/logs/eval/runs/vcap002_dorado_with_trim_hyena/vcap002_dorado_with_trim.fastq_7/predicts/0/ \
	--pdt /projects/b1171/ylk4626/project/DeepChopper/logs/eval/runs/vcap002_dorado_with_trim_hyena/vcap002_dorado_with_trim.fastq_8/predicts/0/ \
	--pdt /projects/b1171/ylk4626/project/DeepChopper/logs/eval/runs/vcap002_dorado_with_trim_hyena/vcap002_dorado_with_trim.fastq_9/predicts/0/ \
	--fq  vcap002_dorado_with_trim.fastq -t 10 -o vcap002_dorado_with_trim_hyena


cargo run --bin predict -r -- \
	--pdt /projects/b1171/ylk4626/project/DeepChopper/logs/eval/runs/vcap_004_cnn_model_cnn/RNA004.fastq_0/predicts/0/ \
	--pdt /projects/b1171/ylk4626/project/DeepChopper/logs/eval/runs/vcap_004_cnn_model_cnn/RNA004.fastq_1/predicts/0/ \
	--pdt /projects/b1171/ylk4626/project/DeepChopper/logs/eval/runs/vcap_004_cnn_model_cnn/RNA004.fastq_2/predicts/0/ \
	--pdt /projects/b1171/ylk4626/project/DeepChopper/logs/eval/runs/vcap_004_cnn_model_cnn/RNA004.fastq_3/predicts/0/ \
	--pdt /projects/b1171/ylk4626/project/DeepChopper/logs/eval/runs/vcap_004_cnn_model_cnn/RNA004.fastq_4/predicts/0/ \
	--pdt /projects/b1171/ylk4626/project/DeepChopper/logs/eval/runs/vcap_004_cnn_model_cnn/RNA004.fastq_5/predicts/0/ \
	--pdt /projects/b1171/ylk4626/project/DeepChopper/logs/eval/runs/vcap_004_cnn_model_cnn/RNA004.fastq_6/predicts/0/ \
	--pdt /projects/b1171/ylk4626/project/DeepChopper/logs/eval/runs/vcap_004_cnn_model_cnn/RNA004.fastq_7/predicts/0/ \
	--pdt /projects/b1171/ylk4626/project/DeepChopper/logs/eval/runs/vcap_004_cnn_model_cnn/RNA004.fastq_8/predicts/0/ \
	--pdt /projects/b1171/ylk4626/project/DeepChopper/logs/eval/runs/vcap_004_cnn_model_cnn/RNA004.fastq_9/predicts/0/ \
	--pdt /projects/b1171/ylk4626/project/DeepChopper/logs/eval/runs/vcap_004_cnn_model_cnn/RNA004.fastq_10/predicts/0/ \
	--pdt /projects/b1171/ylk4626/project/DeepChopper/logs/eval/runs/vcap_004_cnn_model_cnn/RNA004.fastq_11/predicts/0/ \
	--fq data/dorado_without_trim_fqs/RNA004.fastq -t 10 -o vcap_004_cnn

cargo run --bin predict -r -- \
	--pdt /projects/b1171/ylk4626/project/DeepChopper/logs/eval/runs/vcap/VCaP.fastq_0/predicts/0/ \
	--pdt /projects/b1171/ylk4626/project/DeepChopper/logs/eval/runs/vcap/VCaP.fastq_1/predicts/0/ \
	--pdt /projects/b1171/ylk4626/project/DeepChopper/logs/eval/runs/vcap/VCaP.fastq_2/predicts/0/ \
	--pdt /projects/b1171/ylk4626/project/DeepChopper/logs/eval/runs/vcap/VCaP.fastq_3/predicts/0/ \
	--pdt /projects/b1171/ylk4626/project/DeepChopper/logs/eval/runs/vcap/VCaP.fastq_4/predicts/0/ \
	--pdt /projects/b1171/ylk4626/project/DeepChopper/logs/eval/runs/vcap/VCaP.fastq_5/predicts/0/ \
	--pdt /projects/b1171/ylk4626/project/DeepChopper/logs/eval/runs/vcap/VCaP.fastq_6/predicts/0/ \
	--pdt /projects/b1171/ylk4626/project/DeepChopper/logs/eval/runs/vcap/VCaP.fastq_7/predicts/0/ \
	--pdt /projects/b1171/ylk4626/project/DeepChopper/logs/eval/runs/vcap/VCaP.fastq_8/predicts/0/ \
	--pdt /projects/b1171/ylk4626/project/DeepChopper/logs/eval/runs/vcap/VCaP.fastq_9/predicts/0/ \
	--fq data/dorado_without_trim_fqs/VCaP.fastq --ct internal --mcr 70 -t 10 -o vcap_002_hyena_only_internal_mcr_70
