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
  - [4. Predicting Chimeric Reads](#4-predicting-chimeric-reads)
  - [5. Chopping Artificial Sequences](#5-chopping-artificial-sequences)
  - [Next Steps](#next-steps)
  - [Troubleshooting](#troubleshooting)

## Prerequisites

Before we begin, ensure you have the following installed:

- DeepChopper (latest version)
- Dorado (Oxford Nanopore's basecaller)
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
dorado basecaller \
    --model dna_r10.4.1_e8.2_400bps_sup@v4.2.0 \
    --device cuda:all \
    --not_trim \
    path/to/your/pod5/files/ \
    > raw.fastq
```

⚠️ **Important**: Use the `--not_trim` option to preserve potential chimeric sequences.

Replace `path/to/your/pod5/files/` with the directory containing your POD5 files.
The output will be a FASTQ file containing the basecalled sequences.

## 3. Encoding Data with DeepChopper

Prepare your data for the prediction model:

```bash
# Encode the FASTQ file
deepchopper encode raw.fastq
```

For large datasets, use chunking to avoid memory issues:

```bash
deepchopper encode raw.fastq --chunk --chunk-size  100000
```

🔍 **Output**: Look for `encoded_data.parquet` or multiple `.parquet` files if chunking.

## 4. Predicting Chimeric Reads

Analyze the encoded data to identify potential chimeric reads:

```bash
# Predict artifical sequences for reads
deepchopper predict raw.parquet --ouput-path predictions

# Predict artifical sequences for reads using GPU
deepchopper predict raw.parquet --ouput-path predictions --gpus 2
```

For chunked data:

```bash
deepchopper predict raw_chunk1.parquet --output-path predictions_chunk1
deepchopper predict raw_chunk2.parquet --output-path predictions_chunk2
```

📊 **Results**: Check the `predictions` folder for output files.

This step will analyze the encoded data and produce results containing predictions, indicating whether it's likely to be chimeric or not.

## 5. Chopping Artificial Sequences

Remove identified artificial sequences:

```bash
# Chop artificial sequences
deepchopper chop predictions/0 raw.fastq
```

For chunked predictions:

```bash
deepchopper chop predictions_chunk1/0 predictions_chunk2/0 raw.fastq
```

🎉 **Success**: Look for the output file with the `.chop.fq.bgz` suffix.

This command takes the original FASTQ file (`raw.fastq`) and the predictions (`predictions`), and produces a new FASTQ file (with suffix `.chop.fq.bgz`) with the chimeric reads chopped.

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

For more help, visit our [GitHub Issues](https://github.com/ylab-hi/DeepChopper/issues) page.

Happy sequencing, and may your data be artifical-chimera-free! 🧬🔍
