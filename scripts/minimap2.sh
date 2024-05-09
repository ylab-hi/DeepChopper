#!/bin/bash
set -e
set -u
set -o pipefail

if [ $# -ne 1 ]; then
	echo "Usage: $0 <input>"
	exit 1
fi

input=$1

# change extension to .bam from input
bam_output=$(basename $input .fq.gz).bam
bai_output=$(basename $input .fq.gz).bai

echo "Input: $input"
echo "Output: $bam_output"

minimap2 -Y -t 32 -R "@RG\tID:dochoprna002\tSM:hs\tLB:DCPRNA002\tPL:ONT" --MD -ax splice -uf -k14 --junc-bed \
	/projects/b1171/twp7981/database/gencode/gencode.v38.bigbed \
	/projects/b1171/twp7981/database/genome/hg38.fa $input | samtools sort -@ 8 -O BAM -o $bam_output - && samtools index $bam_output $bai_output
