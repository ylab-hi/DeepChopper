# Installation

**Detailed instructions for installing DeepChopper on Linux, macOS, and Windows.**

This guide covers multiple installation methods, GPU setup, and troubleshooting common issues.

## Requirements

- **Python**: 3.10 or later
- **Operating System**: Linux, macOS, or Windows
- **Storage**: At least 2GB free space for the package and models
- **Memory**: Minimum 8GB RAM (16GB+ recommended for large datasets)
- **Optional**: NVIDIA GPU with CUDA support for acceleration

## Quick Installation

The easiest way to install DeepChopper is via pip:

```bash
pip install deepchopper
```

### Verify Installation

Check that DeepChopper is installed correctly:

```bash
deepchopper --help
```

You should see the command-line help information.

## Installation Methods

### Method 1: Using pip (Recommended)

This is the simplest method for most users:

```bash
# Create a virtual environment (recommended)
python -m venv deepchopper_env
source deepchopper_env/bin/activate  # On Windows: deepchopper_env\Scripts\activate

# Install DeepChopper
pip install deepchopper

# Verify installation
deepchopper --version
```

### Method 2: Using conda/mamba

If you prefer conda for package management:

```bash
# Create a new conda environment
conda create -n deepchopper python=3.10
conda activate deepchopper

# Install DeepChopper
pip install deepchopper
```

### Method 3: Development Installation

For developers who want to contribute or modify the source code:

```bash
# Install uv package manager
pip install uv

# Clone the repository
git clone https://github.com/ylab-hi/DeepChopper.git
cd DeepChopper

# Install dependencies
uv sync

# Build and install in development mode
maturin develop --release

# Run tests to verify
uv run pytest tests -k "not slow"
```

For more details, see the [Contributing Guide](contributing.md).

## Platform-Specific Instructions

### Linux

DeepChopper works on most modern Linux distributions:

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install python3.10 python3-pip

# Fedora/CentOS/RHEL
sudo dnf install python3.10 python3-pip

# Install DeepChopper
pip install deepchopper
```

### macOS

```bash
# Install Python using Homebrew (if needed)
brew install python@3.10

# Install DeepChopper
pip install deepchopper
```

!!! note "Apple Silicon (M1/M2/M3)"
    DeepChopper has native support for Apple Silicon Macs. No special configuration needed!

### Windows

```bash
# Open PowerShell or Command Prompt
# Ensure Python 3.10+ is installed

# Create virtual environment
python -m venv deepchopper_env
deepchopper_env\Scripts\activate

# Install DeepChopper
pip install deepchopper
```

## GPU Support

DeepChopper can leverage NVIDIA GPUs for faster processing:

### CUDA Setup

```bash
# Install PyTorch with CUDA support
# Check https://pytorch.org for the latest CUDA-compatible version
pip install torch --index-url https://download.pytorch.org/whl/cu121

# Install DeepChopper
pip install deepchopper

# Verify GPU is available
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### Using GPUs with DeepChopper

```bash
# Use GPU for prediction
deepchopper predict data.parquet --gpus 1

# Use multiple GPUs
deepchopper predict data.parquet --gpus 2
```

## Compatibility Matrix

| Python Version | Linux x86_64 | macOS Intel | macOS Apple Silicon | Windows x86_64 |
|:--------------:|:------------:|:-----------:|:-------------------:|:--------------:|
| 3.10           | ✅           | ✅          | ✅                  | ✅             |
| 3.11           | ✅           | ✅          | ✅                  | ✅             |
| 3.12           | ✅           | ✅          | ✅                  | ✅             |

## Upgrading DeepChopper

To upgrade to the latest version:

```bash
pip install --upgrade deepchopper
```

Check the [changelog](https://github.com/ylab-hi/DeepChopper/blob/main/CHANGELOG.md) for new features and bug fixes.

## Troubleshooting

### Common Issues

#### Issue: `command not found: deepchopper`

**Solution**: Ensure the installation directory is in your PATH:

```bash
# Check where pip installs packages
pip show deepchopper

# Add to PATH (in ~/.bashrc or ~/.zshrc)
export PATH="$HOME/.local/bin:$PATH"
```

#### Issue: Import errors or missing dependencies

**Solution**: Reinstall in a clean virtual environment:

```bash
# Remove old environment
rm -rf deepchopper_env

# Create fresh environment
python -m venv deepchopper_env
source deepchopper_env/bin/activate
pip install --upgrade pip
pip install deepchopper
```

#### Issue: Slow performance without GPU

**Solution**: Install GPU-enabled PyTorch:

```bash
# For CUDA 12.1
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

#### Issue: Out of memory errors

**Solution**: Use chunking for large datasets:

```bash
# Process in smaller chunks
deepchopper chop predictions raw.fq --chunk-size 1000
```

### Additional Prerequisites

Some analyses may require additional tools:

- **Dorado**: For basecalling POD5 files ([installation guide](https://github.com/nanoporetech/dorado))
- **Samtools**: For BAM/FASTQ conversion ([installation guide](http://www.htslib.org/download/))

## Uninstalling

To completely remove DeepChopper:

```bash
pip uninstall deepchopper
```

## Getting Help

If you encounter issues not covered here:

1. Check the [FAQ](faq.md)
2. Search [existing issues](https://github.com/ylab-hi/DeepChopper/issues)
3. [Open a new issue](https://github.com/ylab-hi/DeepChopper/issues/new) with:
   - Your OS and Python version
   - Full error message
   - Steps to reproduce

## Next Steps

- Follow the [Tutorial](tutorial.md) for a complete walkthrough
- Read the [CLI Reference](cli-reference.md) for all available commands
- Check the [FAQ](faq.md) for common questions
