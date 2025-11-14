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
    - [Basic Usage](#basic-usage)
    - [For Large Datasets](#for-large-datasets)
    - [Advanced Options](#advanced-options)
    - [Parameter Guide](#parameter-guide)
  - [4. Predicting Adapter to Detect Artificial Chimeric Reads](#4-predicting-adapter-to-detect-artificial-chimeric-reads)
    - [Basic Usage](#basic-usage-1)
    - [Model Selection](#model-selection)
    - [Advanced Options](#advanced-options-1)
    - [For Chunked Data](#for-chunked-data)
    - [Hardware Acceleration](#hardware-acceleration)
  - [5. Chopping Artificial Sequences](#5-chopping-artificial-sequences)
    - [Basic Usage](#basic-usage-2)
    - [For Chunked Predictions](#for-chunked-predictions)
    - [Advanced Options](#advanced-options-2)
    - [Parameter Guide](#parameter-guide-1)
    - [Understanding the Output](#understanding-the-output)
    - [Performance Notes](#performance-notes)
  - [6. Web Interface (Optional)](#6-web-interface-optional)
  - [Next Steps](#next-steps)
  - [Troubleshooting](#troubleshooting)
    - [Memory Issues](#memory-issues)
    - [Performance Issues](#performance-issues)
    - [Model and Results Issues](#model-and-results-issues)
    - [Hardware and Compatibility Issues](#hardware-and-compatibility-issues)
    - [Getting Help](#getting-help)

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
# Run Dorado without trimming to preserve all sequences
dorado basecaller --no-trim rna002_70bps_hac@v3 200cases.pod5 > raw_no_trim.bam

# Convert BAM to FASTQ
samtools view raw_no_trim.bam -d dx:0 | samtools fastq > raw_no_trim.fastq
```

Replace `200cases.pod5` with the directory containing your POD5 files. Use `rna002_70bps_hac@v3` for RNA002 kit or `rna004_130bps_hac@v5.0.0` for RNA004 kit.

The output will be a FASTQ file containing the basecalled sequences with all adapters preserved for DeepChopper analysis.

📝 **Note**: You can also use Dorado WITH trimming (default behavior without `--no-trim`), then apply DeepChopper. Dorado's trimming removes 3' end adapters, and DeepChopper can identify and remove internal adapter regions that Dorado doesn't detect. Both approaches work well with DeepChopper.

**For convenience**, you can download a pre-prepared FASTQ file for testing:

```bash
wget https://raw.githubusercontent.com/ylab-hi/DeepChopper/refs/heads/main/tests/data/raw_no_trim.fastq
```

## 3. Encoding Data with DeepChopper

Prepare your data for the prediction model:

### Basic Usage

```bash
# Encode the FASTQ file
deepchopper encode raw_no_trim.fastq
```

### For Large Datasets

For large datasets, use chunking to avoid memory issues:

```bash
# Use chunking with default chunk size (1,000,000 reads)
deepchopper encode raw_no_trim.fastq --chunk

# Specify custom chunk size for better memory control
deepchopper encode raw_no_trim.fastq --chunk --chunk-size 100000
```

### Advanced Options

```bash
# Enable verbose output to see progress
deepchopper encode raw_no_trim.fastq --verbose

# Combine chunking with verbose output
deepchopper encode raw_no_trim.fastq --chunk --chunk-size 50000 --verbose
```

### Parameter Guide

- `--chunk, -c`: Enable chunking for large datasets
- `--chunk-size, -s`: Number of reads per chunk (default: 1,000,000)
- `--verbose, -v`: Show detailed progress information

🔍 **Output**:

- Without chunking: `raw_no_trim.parquet`
- With chunking: Multiple `.parquet` files under `raw_no_trim.fq_chunks/` directory

💡 **Tip**: Use chunking for datasets with >1M reads to prevent memory issues.

## 4. Predicting Adapter to Detect Artificial Chimeric Reads

Analyze the encoded data to identify potential chimeric reads:

### Basic Usage

```bash
# Predict artificial sequences for reads (default: RNA002 model)
deepchopper predict raw_no_trim.parquet --output predictions

# Predict artificial sequences for reads using GPU
deepchopper predict raw_no_trim.parquet --output predictions --gpus 2
```

### Model Selection

DeepChopper supports different models optimized for different RNA sequencing kits:

```bash
# Use RNA002 model (default - for RNA002 sequencing kit)
deepchopper predict raw_no_trim.parquet --output predictions --model rna002

# Use RNA004 model (for RNA004 sequencing kit)
deepchopper predict raw_no_trim.parquet --output predictions --model rna004
```

🎯 **Important**: Choose the model that matches your sequencing kit:

- `rna002`: For data generated with the RNA002 sequencing kit
- `rna004`: For data generated with the RNA004 sequencing kit (newer version with improved chemistry)

### Advanced Options

```bash
# Limit the number of samples to process (useful for testing)
deepchopper predict raw_no_trim.parquet --output predictions --max-sample 1000

# Adjust batch size for memory management
deepchopper predict raw_no_trim.parquet --output predictions --batch-size 8

# Combine multiple options
deepchopper predict raw_no_trim.parquet \
    --output predictions \
    --model rna004 \
    --gpus 1 \
    --batch-size 16 \
    --workers 4
```

### For Chunked Data

```bash
deepchopper predict raw_no_trim.fq_chunks/raw_no_trim.fq_0.parquet --output predictions_chunk1
deepchopper predict raw_no_trim.fq_chunks/raw_no_trim.fq_1.parquet --output predictions_chunk2
```

📊 **Results**: Check the `predictions` folder for output files.

This step will analyze the encoded data and produce results containing predictions, indicating whether it's likely to be chimeric or not.

### Hardware Acceleration

DeepChopper automatically detects and uses available hardware acceleration:

- **CUDA GPUs**: Specified with `--gpus N` (where N is the number of GPUs)
- **Apple Silicon (M1/M2/M3)**: Automatically uses Metal Performance Shaders (MPS) when `--gpus 1` is specified
- **CPU**: Default when no GPU is specified or available

## 5. Chopping Artificial Sequences

Remove identified artificial sequences:

### Basic Usage

```bash
# Chop artificial sequences
deepchopper chop predictions/0 raw_no_trim.fastq
```

### For Chunked Predictions

```bash
deepchopper chop predictions_chunk1/0 predictions_chunk2/0 raw_no_trim.fastq
```

### Advanced Options

```bash
# Specify output prefix
deepchopper chop predictions/0 raw_no_trim.fastq --prefix my_cleaned_data

# Control memory usage with batch size (useful for large datasets)
deepchopper chop predictions/0 raw_no_trim.fastq --max-batch 5000

# Adjust smoothing and filtering parameters
deepchopper chop predictions/0 raw_no_trim.fastq \
    --smooth-window 21 \
    --min-interval-size 13 \
    --min-read-length 20

# Include chopped sequences in output
deepchopper chop predictions/0 raw_no_trim.fastq --output-chopped

# Use multiple threads for faster processing
deepchopper chop predictions/0 raw_no_trim.fastq --threads 4
```

### Parameter Guide

Key parameters you can adjust:

- `--prefix, -o`: Custom output file prefix
- `--max-batch`: Maximum batch size for memory management (default: auto)
- `--threads`: Number of threads to use (default: 2)
- `--smooth-window`: Smooth window size for prediction smoothing (default: 21)
- `--min-interval-size`: Minimum interval size to consider (default: 13)
- `--min-read-length`: Minimum read length after chopping (default: 20)
- `--approved-intervals`: Number of approved intervals (default: 20)
- `--output-chopped`: Output the chopped sequences separately
- `--chop-type`: Type of chopping to perform (default: "all")

🎉 **Success**: Look for the output file with the `.chop.fq.bgz` suffix.

This command takes the original FASTQ file (`raw_no_trim.fastq`) and the predictions (`predictions`), and produces a new FASTQ file (with suffix `.chop.fq.bgz`) with the chimeric-artifact chopped.

### Understanding the Output

The default output is a compressed file in BGZIP format:

- **Format**: BGZIP-compressed FASTQ (`.chop.fq.bgz`)
- **View**: Use `zless -S OUTPUT` to view the output file contents in a terminal
- **The `-S` flag**: Prevents line wrapping, making it easier to read long sequences
- **Compatibility**: Can be directly used with most bioinformatics tools that support BGZIP

### Performance Notes

The default parameters used in DeepChopper are optimized based on extensive testing and validation during our research, as detailed in our paper.
These parameters have been shown to provide robust and reliable results across a wide range of sequencing data.

**Processing Time**:

- Demo data: ~20-30 minutes
- Large datasets: May vary depending on:
  - Machine specifications
  - CPU/GPU availability
  - Number of threads used
  - Batch size settings

**Memory Management**:

- Lower batch sizes = less memory but slower processing
- Higher batch sizes = more memory but faster processing

## 6. Web Interface (Optional)

DeepChopper also provides a user-friendly web interface for quick tests and demonstrations:

```bash
# Launch the web interface
deepchopper web
```

This will start a local web server where you can:

- Upload single FASTQ records
- Visualize predictions in real-time
- Test DeepChopper without command-line operations

⚠️ **Note**: The web interface is designed for quick tests with single reads. For production use with large datasets, use the command-line interface.

🌐 **Online Version**: Try DeepChopper online at [Hugging Face Spaces](https://huggingface.co/spaces/yangliz5/deepchopper) without any installation!

## Next Steps

- **Advanced Parameters**: Check [our documentation](./parameters.md) for detailed parameters of the `chop` command
- **CLI Options**: Explore all available options with `deepchopper --help`, `deepchopper encode --help`, `deepchopper predict --help`, etc.
- **Downstream Analysis**: Use your cleaned data for:
  - Transcript annotation
  - Gene expression quantification
  - Gene fusion detection
  - Alternative splicing analysis

## Troubleshooting

### Memory Issues

- **Issue**: Out of memory errors when encoding

  **Solution**:

  - Use the `--chunk` option: `deepchopper encode input.fastq --chunk`
  - Reduce chunk size: `deepchopper encode input.fastq --chunk --chunk-size 50000`
  - Process smaller batches of files if dealing with multiple FASTQ files

- **Issue**: Out of memory errors for CPU or CUDA (GPU) when predicting

  **Solution**:

  - Reduce batch size: `deepchopper predict input.parquet --batch-size 4`
  - Use `--max-sample` for testing: `deepchopper predict input.parquet --max-sample 1000`
  - For large datasets, process in chunks

- **Issue**: Out of memory when chopping

  **Solution**:

  - Reduce the number of threads: `deepchopper chop predictions/0 input.fastq --threads 1`

### Performance Issues

- **Issue**: Slow processing

  **Solution**:

  - Enable GPU acceleration: `deepchopper predict input.parquet --gpus 1`
  - Use chunked encoding for better memory efficiency: `deepchopper encode input.fastq --chunk`
  - Increase threads for chopping: `deepchopper chop predictions/0 input.fastq --threads 4`
  - Increase batch size (if memory allows): `deepchopper predict input.parquet --batch-size 16`

- **Issue**: Apple Silicon (M1/M2/M3) not using GPU

  **Solution**:

  - Specify `--gpus 1` to enable MPS acceleration
  - Ensure PyTorch was installed with MPS support
  - Check with: `python -c "import torch; print(torch.backends.mps.is_available())"`

### Model and Results Issues

- **Issue**: Unexpected results or poor performance

  **Solution**:

  - **Verify model selection**: Use `--model rna002` for RNA002 data or `--model rna004` for RNA004 data
  - **Try both workflows**: You can use Dorado with or without trimming - both work well with DeepChopper
  - Verify input data quality (check FASTQ quality scores)
  - Check DeepChopper version: `deepchopper --version`
  - Review the prediction output files before chopping

### Hardware and Compatibility Issues

- **Issue**: GPU driver compatibility error

  **Solution**:

  - Update your GPU driver to the latest version
  - Install a compatible PyTorch version:
    - CUDA 11.8: `pip install torch --force-reinstall --index-url https://download.pytorch.org/whl/cu118`
    - CUDA 12.1: `pip install torch --force-reinstall --index-url https://download.pytorch.org/whl/cu121`
    - CPU only: `pip install torch --force-reinstall --index-url https://download.pytorch.org/whl/cpu`

- **Issue**: `deepchopper-chop` command not found

  **Solution**:

  - Install the Rust-based chopper: `pip install deepchopper-cli` or follow installation guide
  - Ensure the binary is in your PATH
  - Try reinstalling: `pip install --force-reinstall deepchopper`

### Getting Help

If you encounter issues not covered here:

1. Check the [GitHub Issues](https://github.com/ylab-hi/DeepChopper/issues) for similar problems
2. Review the [full documentation](https://github.com/ylab-hi/DeepChopper)
3. Use `--verbose` flag to see detailed error messages
4. Open a new issue with:
   - DeepChopper version (`deepchopper --version`)
   - Command you ran
   - Full error message
   - System information (OS, Python version, GPU if applicable)

Happy sequencing, and may your data be artifical-chimera-free! 🧬🔍
