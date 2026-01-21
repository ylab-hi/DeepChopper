# CLI Reference

**Complete documentation for all DeepChopper command-line interface commands.**

Reference guide covering all available commands, options, and usage examples.

## Overview

DeepChopper offers several commands for processing Nanopore direct-RNA sequencing data:

- `deepchopper predict` - Detect adapter sequences in FASTQ files
- `deepchopper chop` - Remove detected adapter sequences from reads
- `deepchopper web` - Launch interactive web interface
- `deepchopper --version` - Display version information

## Global Options

These options are available for all commands:

```bash
deepchopper [OPTIONS] COMMAND
```

| Option | Description |
|--------|-------------|
| `--help`, `-h` | Show help message and exit |
| `--version` | Show version and exit |

## Commands

### predict

Detect adapter sequences and chimeric artifacts in sequencing data.

```bash
deepchopper predict [OPTIONS] DATA_PATH
```

#### Arguments

| Argument | Type | Required | Description |
|----------|------|----------|-------------|
| `DATA_PATH` | Path | Yes | Path to FASTQ or Parquet file |

#### Options

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--gpus` | `-g` | Integer | 0 | Number of GPUs to use (0 for CPU) |
| `--output` | `-o` | Path | None | Output directory for predictions |
| `--batch-size` | `-b` | Integer | 12 | Batch size for processing |
| `--workers` | `-w` | Integer | 0 | Number of data loader workers |
| `--model` | `-m` | String | rna002 | Model to use (rna002 or rna004) |
| `--limit-batches` | | Integer | None | Limit number of batches to process |
| `--max-sample` | | Integer | None | Maximum number of samples to process |
| `--verbose` | `-v` | Flag | False | Enable verbose output |

#### Examples

**Basic prediction (CPU):**

```bash
deepchopper predict raw_reads.fastq
```

**Using GPU acceleration:**

```bash
deepchopper predict raw_reads.fastq --gpus 1
```

**Using RNA004 model:**

```bash
deepchopper predict raw_reads.fastq --model rna004
```

**Custom output location:**

```bash
deepchopper predict raw_reads.fastq --output ./predictions
```

**Process a subset for testing:**

```bash
deepchopper predict raw_reads.fastq --max-sample 1000
```

**Adjust batch size for memory:**

```bash
deepchopper predict raw_reads.fastq --batch-size 32 --gpus 1
```

#### Output

Creates a directory containing:

- `predictions/` - Directory containing predicted adapter positions for each read

---

### chop

Remove adapter sequences from reads based on predictions.

```bash
deepchopper chop [OPTIONS] PREDICTIONS_DIR FASTQ_FILE
```

#### Arguments

| Argument | Type | Required | Description |
|----------|------|----------|-------------|
| `PREDICTIONS_DIR` | Path | Yes | Directory containing predictions from `predict` command |
| `FASTQ_FILE` | Path | Yes | Original FASTQ file to chop |

#### Options

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--output` | `-o` | Path | Auto | Output FASTQ file path |
| `--chunk-size` | | Integer | 10000 | Number of reads to process per chunk |
| `--min-read-length` | | Integer | 20 | Minimum length of chopped reads to keep |
| `--smooth-window` | | Integer | 21 | Sliding window size for smoothing |
| `--min-interval-size` | | Integer | 13 | Minimum adapter region size |
| `--max-process-intervals` | | Integer | 4 | Maximum adapter regions per read |
| `--verbose` | `-v` | Flag | False | Enable verbose output |

#### Examples

**Basic chopping:**

```bash
deepchopper chop predictions/ raw_reads.fastq
```

**Custom output file:**

```bash
deepchopper chop predictions/ raw_reads.fastq --output chopped.fastq
```

**Memory-efficient processing:**

```bash
deepchopper chop predictions/ raw_reads.fastq --chunk-size 1000
```

**High-performance processing:**

```bash
deepchopper chop predictions/ raw_reads.fastq --chunk-size 50000
```

**Adjust sensitivity:**

```bash
deepchopper chop predictions/ raw_reads.fastq --smooth-window 31 --min-interval-size 15
```

**Keep only longer reads:**

```bash
deepchopper chop predictions/ raw_reads.fastq --min-read-length 50
```

#### Output

Creates:

- Chopped FASTQ file with adapter sequences removed
- Reads split at adapter positions
- Statistics about processing

#### Memory Usage Guidelines

| Chunk Size | Memory Usage | Speed | Use Case |
|------------|--------------|-------|----------|
| 1,000 | ~1-2 GB | Slower | Memory-constrained systems |
| 10,000 (default) | ~5-10 GB | Balanced | General use |
| 50,000 | ~20-50 GB | Fastest | High-memory systems |

---

### web

Launch interactive web interface for DeepChopper.

```bash
deepchopper web [OPTIONS]
```

#### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--port` | Integer | 7860 | Port to run the web server on |
| `--share` | Flag | False | Create public Gradio link |

#### Examples

**Start web interface:**

```bash
deepchopper web
```

**Custom port:**

```bash
deepchopper web --port 8080
```

**Create shareable link:**

```bash
deepchopper web --share
```

#### Note

The web interface is limited to processing one FASTQ record at a time. For batch processing, use the `predict` and `chop` commands.

---

## Complete Workflow Examples

### Basic Workflow

```bash
# 1. Predict adapters
deepchopper predict raw_reads.fastq --output predictions

# 2. Chop reads
deepchopper chop predictions/ raw_reads.fastq --output chopped.fastq
```

### GPU-Accelerated Workflow

```bash
# Use GPU for faster prediction
deepchopper predict raw_reads.fastq --gpus 1 --batch-size 32 --output predictions

# Chop with default settings
deepchopper chop predictions/ raw_reads.fastq --output chopped.fastq
```

### Memory-Efficient Workflow

```bash
# Process in smaller batches
deepchopper predict raw_reads.fastq --batch-size 8 --output predictions

# Chop with small chunks
deepchopper chop predictions/ raw_reads.fastq --chunk-size 1000 --output chopped.fastq
```

### RNA004 Workflow

```bash
# Use RNA004 model
deepchopper predict raw_reads.fastq --model rna004 --output predictions

# Adjust parameters for RNA004
deepchopper chop predictions/ raw_reads.fastq \
  --smooth-window 31 \
  --min-interval-size 15 \
  --output chopped.fastq
```

### Testing Workflow

```bash
# Test on subset of data
deepchopper predict raw_reads.fastq --max-sample 100 --output test_predictions

# Check results
deepchopper chop test_predictions/ raw_reads.fastq --output test_chopped.fastq
```

## Tips and Best Practices

### Performance Optimization

1. **Use GPU when available**: Adds `--gpus 1` for 10-50x speedup
2. **Adjust batch size**: Larger batches are faster but use more memory
3. **Use appropriate chunk size**: Balance between memory and speed for chopping
4. **Process in parallel**: Run multiple instances on different files

### Quality Control

1. **Inspect predictions**: Check `predictions/` directory for adapter positions
2. **Monitor statistics**: Review processing logs for quality metrics
3. **Validate results**: Compare input/output read counts
4. **Adjust parameters**: Fine-tune based on your data characteristics

### Troubleshooting

- **Out of memory**: Reduce `--batch-size` or `--chunk-size`
- **Slow processing**: Enable GPU with `--gpus 1`
- **Too many fragments**: Increase `--smooth-window` or reduce `--max-process-intervals`
- **Missed adapters**: Decrease `--min-interval-size` or `--smooth-window`

## See Also

- [Tutorial](tutorial.md) - Step-by-step guide with real data
- [Parameters Guide](parameters.md) - Detailed parameter optimization
- [Installation](installation.md) - Setup instructions
- [FAQ](faq.md) - Common questions and answers
