#!/bin/bash -l
#SBATCH --job-name=dc_cnn
#SBATCH --output=dc_cnn_%j.log
#SBATCH --nodes=1
#SBATCH --time=48:00:00
#SBATCH --mem=30gb
#SBATCH --ntasks=16
#SBATCH --account=p31888
#SBATCH -p gengpu
#SBATCH --gres=gpu:a100:2

cd $SLURM_SUBMIT_DIR
date
hostname
pwd

set -o nounset
set -o errexit
set -x

echo "Starting job $SLURM_JOBID"
nvidia-smi

poe train  experiment=cnn
