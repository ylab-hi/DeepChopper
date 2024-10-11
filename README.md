# <img src="./documentation/logo.webp" alt="logo" height="100"/> **DeepChopper** [![social](https://img.shields.io/github/stars/ylab-hi/DeepChopper?style=social)](https://github.com/ylab-hi/DeepChopper/stargazers)

[![pypi](https://img.shields.io/pypi/v/deepchopper.svg)](https://pypi.python.org/pypi/deepchopper)
[![license](https://img.shields.io/pypi/l/deepchopper.svg)](https://github.com/ylab-hi/DeepChopper/blob/main/LICENSE)
[![pypi version](https://img.shields.io/pypi/pyversions/deepchopper.svg)](https://pypi.python.org/pypi/deepbiop)
[![Actions status](https://github.com/ylab-hi/DeepChopper/actions/workflows/release-python.yml/badge.svg)](https://github.com/ylab-hi/DeepChopper/actions)
[![platform](https://img.shields.io/badge/platform-linux%20%7C%20osx%20%7C%20win-blue)](https://pypi.org/project/deepchopper/#files)
[![Space](https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-hf-spaces-md.svg)](https://huggingface.co/spaces/yangliz5/deepchopper)

<!--toc:start-->

- [ **DeepChopper** ](#-deepchopper-)
  - [Quick Start: Try DeepChopper Online](#quick-start-try-deepchopper-online)
  - [Install](#install)
  - [Usage](#usage)
    - [Command-Line Interface](#command-line-interface)
    - [Library](#library)
  - [Cite](#cite)
  - [🤜 Contribution](#-contribution)
    - [Build Environment](#build-environment)
    - [Install Dependencies](#install-dependencies)

<!--toc:end-->

DeepChopper leverages language model to accurately detect and chop artificial sequences which may cause chimeric reads, ensuring higher quality and more reliable sequencing results.
By integrating seamlessly with existing workflows, DeepChopper provides a robust solution for researchers and bioinformatics working with NanoPore direct-RNA sequencing data.

## Quick Start: Try DeepChopper Online

Experience DeepChopper instantly through our user-friendly web interface. No installation required!

Simply click the button below to launch the web application and start exploring DeepChopper's capabilities:

[![Open in Hugging Face Spaces](https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-hf-spaces-md.svg)](https://huggingface.co/spaces/yangliz5/deepchopper)

This online version provides a convenient way to:

- Upload your sequencing data
- Run DeepChopper's analysis
- Visualize results
- Experiment with different parameters

It's perfect for quick tests or when you want to showcase DeepChopper's functionality without local setup.
However, for more extensive analyses or custom workflows, we recommend installing DeepChopper on your machine.
Because the online version is limited to one FASTQ record at a time, it may not be suitable for large-scale projects.

## Install

DeepChopper can be installed using pip, the Python package installer. Follow these steps to install:

1. Ensure you have Python 3.10 or later installed on your system.

2. It's recommended to create a virtual environment:

   ```bash
   python -m venv deepchopper_env
   source deepchopper_env/bin/activate  # On Windows use `deepchopper_env\Scripts\activate`
   ```

3. Install DeepChopper:

   ```bash
   pip install deepchopper
   ```

4. Verify the installation:

   ```bash
   deepchopper --help
   ```

Note: If you encounter any issues, please check our GitHub repository for troubleshooting guides or to report a problem.

## Usage

We provide a [complete guide](./documentation/tutorial.md) on how to use DeepChopper for NanoPore direct-RNA sequencing data.
Below is a brief overview of the command-line interface and library usage.

### Command-Line Interface

DeepChopper provides a command-line interface (CLI) for easy access to its features. In total, there are three commands: `encode`, `predict`, and `chop`.
DeepChopper can be used to encode, predict, and chop chimeric reads in direct-RNA sequencing data.

Firstly, we need to encode the input data using the `encode` command, which will generate a `.parquet` file.

```bash
deepchopper endcode <input.fq>
```

Next, we can use the `predict` command to predict chimeric reads in the encoded data.

```bash
deepchopper predict <input.parquet> --ouput-path predictions
```

If you have GPUs, you can use the `--gpus` flag to specify the GPU device.

```bash
deepchopper predict <input.parquet> --ouput-path predictions --gpus 2
```

Finally, we can use the `chop` command to chop the chimeric reads in the input data.

```bash
deepchopper chop <predictions> raw.fq
```

Besides, DeepChopper provides a web-based user interface for users to interact with the tool.
However, the web-based application can only take one FASTQ record at a time.

```bash
deepchopper web
```

### Library

```python
import deepchopper

model = deepchopper.DeepChopper.from_pretrained("yangliz5/deepchopper")
```

## Cite

If you use DeepChopper in your research, please cite the following paper:

```bibtex

```

## 🤜 Contribution

### Build Environment

```bash
git clone https://github.com/ylab-hi/DeepChopper.git
cd DeepChopper
conda env create -n environment.yaml
conda activate deepchopper
```

### Install Dependencies

```bash
pip install pipx
pipx install --suffix @master git+https://github.com/python-poetry/poetry.git@master
poetry@master install
```
