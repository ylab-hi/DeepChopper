# FAQ

**Frequently asked questions and answers about DeepChopper.**

Find quick answers to common questions about installation, usage, performance, and troubleshooting.

## General Questions

### What is DeepChopper?

DeepChopper is a deep learning tool designed to detect and remove chimeric artifacts in Nanopore direct RNA sequencing data. It uses a transformer-based language model to identify adapter sequences within base-called reads that traditional basecallers miss.

### Why do I need DeepChopper?

Chimeric reads (reads containing internal adapter sequences) can lead to:

- False gene fusion calls
- Incorrect transcript annotations
- Inflated gene expression estimates
- Poor transcriptome assembly quality

DeepChopper removes these artifacts, improving downstream analysis accuracy.

### How does DeepChopper differ from Dorado's trimming?

- **Dorado**: Trims adapters from read ends (5' and 3')
- **DeepChopper**: Detects and removes **internal** adapter sequences that Dorado misses

You can use both tools together for comprehensive adapter removal.

## Installation & Setup

### What are the system requirements?

**Minimum:**
- Python 3.10+
- 8GB RAM
- 2GB storage

**Recommended:**
- 16GB+ RAM
- NVIDIA GPU with CUDA support
- 10GB storage (for models and data)

### Can I use DeepChopper without a GPU?

Yes! DeepChopper works on CPU, though it's slower. For small datasets (<10K reads), CPU processing is reasonable. For larger datasets, GPU acceleration is recommended.

### Which operating systems are supported?

DeepChopper supports:
- Linux (x86_64)
- macOS (Intel and Apple Silicon)
- Windows (x86_64)

Pre-built wheels are available on PyPI for all platforms.

## Usage Questions

### Which model should I use: rna002 or rna004?

- **RNA002**: For data sequenced with RNA002 chemistry
- **RNA004**: For data sequenced with RNA004 or newer chemistries

DeepChopper has zero-shot capability, so the RNA002 model works well on RNA004 data, but using the matching model is recommended for best results.

### How long does processing take?

Processing time depends on:
- Dataset size
- Hardware (CPU vs GPU)
- Batch size

**Approximate times for 1 million reads:**
- CPU: 2-6 hours
- GPU (single): 10-30 minutes
- GPU (multiple): 5-15 minutes

### Can I process multiple files in parallel?

Yes! You can run multiple DeepChopper instances simultaneously:

```bash
# Process multiple files
for file in *.fastq; do
  deepchopper predict "$file" --output "predictions_${file%.fastq}" &
done
wait
```

### What input formats are supported?

- **FASTQ** (`.fastq`, `.fq`, `.fastq.gz`, `.fq.gz`)
- **Parquet** (for already-encoded data)

### What is the output format?

- **Predictions**: Parquet files with adapter positions
- **Chopped reads**: FASTQ format with adapters removed

## Performance & Optimization

### How can I speed up processing?

1. **Use GPU**: Add `--gpus 1` to prediction
2. **Increase batch size**: Try `--batch-size 32` or higher
3. **Use multiple GPUs**: `--gpus 2` for parallel processing
4. **Process in parallel**: Run multiple instances on different files

### I'm running out of memory. What should I do?

**For prediction:**
```bash
# Reduce batch size
deepchopper predict data.fastq --batch-size 4
```

**For chopping:**
```bash
# Reduce chunk size
deepchopper chop predictions/ data.fastq --chunk-size 1000
```

### How much memory do I need?

| Dataset Size | Prediction (CPU) | Prediction (GPU) | Chopping |
|-------------|------------------|------------------|----------|
| 100K reads | 2-4 GB | 4-6 GB | 1-2 GB |
| 1M reads | 4-8 GB | 8-12 GB | 2-5 GB |
| 10M reads | 8-16 GB | 12-24 GB | 5-20 GB |

## Results & Quality

### How do I know if DeepChopper worked correctly?

Check these indicators:

1. **Output file size**: Should be smaller than input (adapters removed)
2. **Read count**: May increase (reads split at adapters)
3. **Log messages**: Look for "processed X reads" messages
4. **Quality metrics**: Review prediction confidence scores

### Why are there more reads in the output?

This is expected! DeepChopper splits chimeric reads at adapter positions, creating multiple valid reads from single chimeric reads.

**Example:**
```
Input:  1 chimeric read (Read1-Adapter-Read2)
Output: 2 valid reads (Read1, Read2)
```

### How do I validate the results?

1. **Alignment improvement**: Map to reference genome, check alignment rates
2. **Chimeric alignment reduction**: Count chimeric alignments before/after
3. **Gene fusion validation**: Verify gene fusion calls are more accurate
4. **Visual inspection**: Use web interface to inspect individual reads

### Can DeepChopper introduce false positives?

Yes, like any tool, false positives are possible. To reduce them:

- Increase `--smooth-window` (e.g., 31)
- Increase `--min-interval-size` (e.g., 15)
- Use the model matching your chemistry

## Troubleshooting

### Error: "command not found: deepchopper"

**Solution**: Ensure DeepChopper is installed and in your PATH:

```bash
# Check installation
pip show deepchopper

# Add to PATH if needed
export PATH="$HOME/.local/bin:$PATH"
```

### Error: "CUDA out of memory"

**Solution**: Reduce batch size:

```bash
deepchopper predict data.fastq --gpus 1 --batch-size 8
```

### Error: "FileNotFoundError"

**Solution**: Check file paths and ensure files exist:

```bash
# Verify file exists
ls -lh data.fastq

# Use absolute paths
deepchopper predict /full/path/to/data.fastq
```

### Predictions are empty or incorrect

**Possible causes:**

1. **Wrong model**: Make sure you're using the correct model (rna002 vs rna004)
2. **Already trimmed data**: If Dorado already trimmed adapters, DeepChopper may not find internal adapters
3. **Low-quality data**: Very noisy data may produce poor predictions

**Solutions:**

- Use matching model for your chemistry
- If data is already clean, DeepChopper may not be needed
- Adjust parameters (see [Parameters Guide](parameters.md))

### Processing is very slow

**Common causes:**

1. **Using CPU instead of GPU**
2. **Small batch size**
3. **Many workers with limited CPU cores**

**Solutions:**

```bash
# Enable GPU
deepchopper predict data.fastq --gpus 1 --batch-size 32

# Optimize workers (usually 0 or 4 works best)
deepchopper predict data.fastq --workers 4
```

## Advanced Usage

### Can I train my own model?

Yes! DeepChopper is built on PyTorch Lightning. See the [development documentation](contributing.md) for training instructions.

### Can I use DeepChopper programmatically?

Yes! DeepChopper provides a Python API. However, for most use cases, the CLI is recommended as it's optimized and easier to use.

### Does DeepChopper work with DNA sequencing?

DeepChopper is specifically designed and trained for direct RNA sequencing. It may not work well with DNA data.

### Can I use DeepChopper with other sequencing platforms?

DeepChopper is optimized for Oxford Nanopore direct RNA sequencing. It's not tested or recommended for other platforms (Illumina, PacBio, etc.).

## Data & Privacy

### Does DeepChopper send my data anywhere?

No. DeepChopper processes all data locally. The only network access is for:
- Downloading models from Hugging Face Hub (one-time)
- Using the optional `--share` flag in web interface

### Where are models stored?

Models are cached locally in:
- Linux: `~/.cache/huggingface/`
- macOS: `~/Library/Caches/huggingface/`
- Windows: `%USERPROFILE%\.cache\huggingface\`

### Can I use DeepChopper offline?

Yes, after the initial model download. Models are cached locally and don't require internet access for subsequent runs.

## Getting More Help

### Where can I find more information?

- [Tutorial](tutorial.md) - Complete walkthrough
- [CLI Reference](cli-reference.md) - All commands and options
- [Parameters Guide](parameters.md) - Optimization tips
- [GitHub Issues](https://github.com/ylab-hi/DeepChopper/issues) - Report bugs
- [GitHub Discussions](https://github.com/ylab-hi/DeepChopper/discussions) - Ask questions

### How do I report a bug?

1. Check [existing issues](https://github.com/ylab-hi/DeepChopper/issues)
2. [Open a new issue](https://github.com/ylab-hi/DeepChopper/issues/new) with:
   - DeepChopper version (`deepchopper --version`)
   - Operating system and Python version
   - Full error message
   - Steps to reproduce

### How do I request a feature?

Open a [GitHub Discussion](https://github.com/ylab-hi/DeepChopper/discussions) describing:
- The feature you'd like
- Your use case
- Why it would be helpful

### How can I contribute?

We welcome contributions! See the [Contributing Guide](contributing.md) for:
- Setting up development environment
- Code style guidelines
- Testing procedures
- Pull request process
