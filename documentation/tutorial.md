# Tutorial: Using DeepChopper for Nanopore Direct-RNA Sequencing Data Analysis

Welcome to the DeepChopper tutorial! This guide will walk you through the process of identifying and removing chimeric artificial reads in Nanopore direct-RNA sequencing data.
Whether you're new to bioinformatics or an experienced researcher, this tutorial will help you get the most out of DeepChopper.

## Table of Contents

- [Tutorial: Using DeepChopper for Nanopore Direct-RNA Sequencing Data Analysis](#tutorial-using-deepchopper-for-nanopore-direct-rna-sequencing-data-analysis)
  - [Table of Contents](#table-of-contents)
  - [Prerequisites](#prerequisites)
  - [1. Data Acquisition](#1-data-acquisition)
  - [2. Basecall Using Dorado](#2-basecall-using-dorado)
  - [3. Encoding Data with DeepChopper](#3-encoding-data-with-deepchopper)
  - [4. Predicting Adapter to Detect Artificial Chimeric Reads](#4-predicting-adapter-to-detect-artificial-chimeric-reads)
  - [5. Chopping Artificial Sequences](#5-chopping-artificial-sequences)
  - [Next Steps](#next-steps)
  - [Troubleshooting](#troubleshooting)

## Prerequisites

Before we begin, ensure you have the following installed:

- DeepChopper (latest version)
- Dorado (Oxford Nanopore's basecaller)
- Samtools (for BAM to FASTQ conversion)
- Sufficient storage space for Nanopore data

## 1. Data Acquisition

Start by obtaining your Nanopore direct-RNA sequencing data (POD5 files).

```bash
# Example: Download sample data (replace with your actual data source)
wget https://raw.githubusercontent.com/ylab-hi/DeepChopper/refs/heads/main/tests/data/200cases.pod5
```

💡 **Tip**: Organize your data in a dedicated project folder for easy management.

## 2. Basecall Using Dorado

Convert raw signal data to nucleotide sequences using Dorado.

```bash
# Install Dorado (if not already installed)
# Run Dorado without trimming
dorado basecaller --no-trim rna002_70bps_hac@v3 200cases.pod5 > raw_no_trim.bam

# Convert BAM to FASTQ
samtools view raw_no_trim.bam -d dx:0 | samtools fastq > raw_no_trim.fastq
```

⚠️ **Important**: Use the `--not_trim` option to preserve potential chimeric sequences.

Replace `200cases.pod5` with the directory containing your POD5 files.
The output will be a FASTQ file containing the basecalled sequences.

**Note**: For convenience, you can download a pre-prepared FASTQ file directly:

```bash
wget https://raw.githubusercontent.com/ylab-hi/DeepChopper/refs/heads/main/tests/data/raw_no_trim.fastq
```

## 3. Encoding Data with DeepChopper

Prepare your data for the prediction model:

```bash
# Encode the FASTQ file
deepchopper encode raw_no_trim.fastq
```

For large datasets, use chunking to avoid memory issues:

```bash
deepchopper encode raw_no_trim.fastq --chunk --chunk-size  100000
```

🔍 **Output**: Look for `raw_no_trim.parquet` or multiple `.parquet` files under `raw_no_trim.fq_chunks` if chunking.

## 4. Predicting Adapter to Detect Artificial Chimeric Reads

Analyze the encoded data to identify potential chimeric reads:

```bash
# Predict artifical sequences for reads
deepchopper predict raw_no_trim.parquet --ouput predictions

# Predict artifical sequences for reads using GPU
deepchopper predict raw_no_trim.parquet --ouput predictions --gpus 2
```

For chunked data:

```bash
deepchopper predict raw_no_trim.fq_chunks/raw_no_trim.fq_0.parquet --output predictions_chunk1
deepchopper predict raw_no_trim.fq_chunks/raw_no_trim.fq_1.parquet --output predictions_chunk2
```

📊 **Results**: Check the `predictions` folder for output files.

This step will analyze the encoded data and produce results containing predictions, indicating whether it's likely to be chimeric or not.

## 5. Chopping Artificial Sequences

Remove identified artificial sequences:

```bash
# Chop artificial sequences
deepchopper chop predictions/0 raw_no_trim.fastq
```

For chunked predictions:

```bash
deepchopper chop predictions_chunk1/0 predictions_chunk2/0 raw_no_trim.fastq
```

🎉 **Success**: Look for the output file with the `.chop.fq.bgz` suffix.

This command takes the original FASTQ file (`raw_no_trim.fastq`) and the predictions (`predictions`), and produces a new FASTQ file (with suffix `.chop.fq.bgz`) with the chimeric reads chopped.

## Next Steps

- Explore advanced DeepChopper options with `deepchopper --help`
- Use your cleaned data for downstream analyses
- Check our documentation for integration with other bioinformatics tools

## Troubleshooting

- **Issue**: Out of memory errors

  **Solution**: Try using the `--chunk` option in the encode step

- **Issue**: Slow processing

  **Solution**: Ensure you're using GPU acceleration if available

- **Issue**: Unexpected results

  **Solution**: Verify input data quality and check DeepChopper version

- **Issue**: GPU driver compatibility error

  **Solution**: Update your GPU driver or install a compatible PyTorch version e.g., `pip install torch --force-reinstall --index-url https://download.pytorch.org/whl/cu118` to install a CUDA 11.8 compatible version.

For more help, visit our [GitHub Issues](https://github.com/ylab-hi/DeepChopper/issues) page.

Happy sequencing, and may your data be artifical-chimera-free! 🧬🔍
