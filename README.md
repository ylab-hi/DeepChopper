# <img src="./documentation/logo.webp" alt="logo" height="100"/> **DeepChopper** [![social](https://img.shields.io/github/stars/ylab-hi/DeepChopper?style=social)](https://github.com/ylab-hi/DeepChopper/stargazers)

[![pypi](https://img.shields.io/pypi/v/deepchopper.svg)](https://pypi.python.org/pypi/deepchopper)
[![license](https://img.shields.io/pypi/l/deepchopper.svg)](https://github.com/ylab-hi/DeepChopper/blob/main/LICENSE)
[![pypi version](https://img.shields.io/pypi/pyversions/deepchopper.svg)](https://pypi.python.org/pypi/deepbiop)
[![Actions status](https://github.com/ylab-hi/DeepChopper/actions/workflows/release-python.yml/badge.svg)](https://github.com/ylab-hi/DeepChopper/actions)
[![platform](https://img.shields.io/badge/platform-linux%20%7C%20osx%20%7C%20win-blue)](https://pypi.org/project/deepchopper/#files)
[![Space](https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-hf-spaces-md.svg)](https://huggingface.co/spaces/yangliz5/deepchopper)

<!--toc:start-->

- [ **DeepChopper** ](#-deepchopper-)
  - [🚀 Quick Start: Try DeepChopper Online](#-quick-start-try-deepchopper-online)
  - [📦 Installation](#-installation)
  - [🛠️ Usage](#️-usage)
    - [Command-Line Interface](#command-line-interface)
    - [Python Library](#python-library)
  - [📚 Cite](#-cite)
  - [🤝 Contribution](#-contribution)
    - [Build Environment](#build-environment)
    - [Install Dependencies](#install-dependencies)
  - [📬 Support](#-support)

<!--toc:end-->

🧬 DeepChopper leverages language model to accurately detect and chop artificial sequences which may cause chimeric reads, ensuring higher quality and more reliable sequencing results.
By integrating seamlessly with existing workflows, DeepChopper provides a robust solution for researchers and bioinformatics working with NanoPore direct-RNA sequencing data.

## 🚀 Quick Start: Try DeepChopper Online

Experience DeepChopper instantly through our user-friendly web interface. No installation required!
Simply click the button below to launch the web application and start exploring DeepChopper's capabilities:

[![Open in Hugging Face Spaces](https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-hf-spaces-md.svg)](https://huggingface.co/spaces/yangliz5/deepchopper)

**What you can do online:**

- 📤 Upload your sequencing data
- 🔬 Run DeepChopper's analysis
- 📊 Visualize results
- 🎛️ Experiment with different parameters

Perfect for quick tests or demonstrations! However, for extensive analyses or custom workflows, we recommend installing DeepChopper locally.

> ⚠️ Note: The online version is limited to one FASTQ record at a time and may not be suitable for large-scale projects.

## 📦 Installation

DeepChopper can be installed using pip, the Python package installer.
Follow these steps to install:

1. Ensure you have Python 3.10 or later installed on your system.

2. Create a virtual environment (recommended):

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

🆘 Trouble installing? Check our [Troubleshooting Guide](./docs/troubleshooting.md) or [open an issue](https://github.com/ylab-hi/DeepChopper/issues).

## 🛠️ Usage

For a comprehensive guide, check out our [full tutorial](./documentation/tutorial.md).
Here's a quick overview:

### Command-Line Interface

DeepChopper offers three main commands: `encode`, `predict`, and `chop`.

1. **Encode** your input data:

   ```bash
   deepchopper encode <input.fq>
   ```

2. **Predict** chimeric reads:

   ```bash
   deepchopper predict <input.parquet> --output-path predictions
   ```

   Using GPUs? Add the `--gpus` flag:

   ```bash
   deepchopper predict <input.parquet> --output-path predictions --gpus 2
   ```

3. **Chop** the chimeric reads:

   ```bash
   deepchopper chop <predictions> raw.fq
   ```

Want a GUI? Launch the web interface (note: limited to one FASTQ record at a time):

```bash
deepchopper web
```

### Python Library

Integrate DeepChopper into your Python scripts:

```python
import deepchopper

model = deepchopper.DeepChopper.from_pretrained("yangliz5/deepchopper")
# Your analysis code here
```

## 📚 Cite

If DeepChopper aids your research, please cite our paper:

```bibtex

```

## 🤝 Contribution

We welcome contributions! Here's how to set up your development environment:

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

🎉 Ready to contribute? Check out our [Contribution Guidelines](./CONTRIBUTING.md) to get started!

## 📬 Support

Need help? Have questions?

- 📖 Check our [Documentation](./docs)
- 💬 Join our [Community Forum](https://github.com/ylab-hi/DeepChopper/discussions)
- 🐛 [Report issues](https://github.com/ylab-hi/DeepChopper/issues)

---

DeepChopper is developed with ❤️ by the YLab team.
Happy sequencing! 🧬🔬
