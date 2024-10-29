# <img src="./documentation/logo.webp" alt="logo" height="100"/> **DeepChopper** [![social](https://img.shields.io/github/stars/ylab-hi/DeepChopper?style=social)](https://github.com/ylab-hi/DeepChopper/stargazers)

[![pypi](https://img.shields.io/pypi/v/deepchopper.svg)](https://pypi.python.org/pypi/deepchopper)
[![PyPI - Wheel](https://img.shields.io/pypi/wheel/deepchopper)](https://pypi.org/project/deepchopper/#files)
[![license](https://img.shields.io/pypi/l/deepchopper.svg)](https://github.com/ylab-hi/DeepChopper/blob/main/LICENSE)
[![pypi version](https://img.shields.io/pypi/pyversions/deepchopper.svg)](https://pypi.python.org/pypi/deepbiop)
[![platform](https://img.shields.io/badge/platform-linux%20%7C%20osx%20%7C%20win-blue)](https://pypi.org/project/deepchopper/#files)
[![Actions status](https://github.com/ylab-hi/DeepChopper/actions/workflows/release-python.yml/badge.svg)](https://github.com/ylab-hi/DeepChopper/actions)
[![Space](https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-hf-spaces-md.svg)](https://huggingface.co/spaces/yangliz5/deepchopper)

<!--toc:start-->

- [ **DeepChopper** ](#-deepchopper-)
  - [🚀 Quick Start: Try DeepChopper Online](#-quick-start-try-deepchopper-online)
  - [📦 Installation](#-installation)
    - [Compatibility and Support](#compatibility-and-support)
      - [PyPI Support](#pypi-support)
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

### Compatibility and Support

DeepChopper is designed to work across various platforms and Python versions.
Below are the compatibility matrices for PyPI installations:

#### [PyPI Support][pypi]

| Python Version | Linux x86_64 | macOS Intel | macOS Apple Silicon | Windows x86_64 |
| :------------: | :----------: | :---------: | :-----------------: | :------------: |
|      3.10      |      ✅      |     ✅      |         ✅          |       ✅       |
|      3.11      |      ✅      |     ✅      |         ✅          |       ✅       |
|      3.12      |      ✅      |     ✅      |         ✅          |       ✅       |

🆘 Trouble installing? Check our [Troubleshooting Guide](https://github.com/ylab-hi/DeepChopper/blob/main/documentation/tutorial.md#troubleshooting) or [open an issue](https://github.com/ylab-hi/DeepChopper/issues).

## 🛠️ Usage

For a comprehensive guide, check out our [full tutorial](./documentation/tutorial.md).
Here's a quick overview:

### Command-Line Interface

DeepChopper offers three main commands: `encode`, `predict`, and `chop`.

1. **Encode** your input data:

   ```bash
   deepchopper encode <input.fq>
   ```

2. **Predict** chimera artifacts:

   ```bash
   deepchopper predict <input.parquet> --output predictions
   ```

   Using GPUs? Add the `--gpus` flag:

   ```bash
   deepchopper predict <input.parquet> --output predictions --gpus 2
   ```

3. **Chop** chimera artifacts:

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

If DeepChopper aids your research, please cite [our paper](https://www.biorxiv.org/content/10.1101/2024.10.23.619929v2):

```bibtex
@article {Li2024.10.23.619929,
        author = {Li, Yangyang and Wang, Ting-You and Guo, Qingxiang and Ren, Yanan and Lu, Xiaotong and Cao, Qi and Yang, Rendong},
        title = {A Genomic Language Model for Chimera Artifact Detection in Nanopore Direct RNA Sequencing},
        elocation-id = {2024.10.23.619929},
        year = {2024},
        doi = {10.1101/2024.10.23.619929},
        publisher = {Cold Spring Harbor Laboratory},
        abstract = {Chimera artifacts in nanopore direct RNA sequencing (dRNA-seq) data can confound transcriptome analyses, yet no existing tools are capable of detecting and removing them due to limitations in basecalling models. We present DeepChopper, a genomic language model that accurately identifies and eliminates adapter sequences within base-called dRNA-seq reads, effectively removing chimeric read artifacts. DeepChopper significantly improves critical downstream analyses, including transcript annotation and gene fusion detection, enhancing the reliability and utility of nanopore dRNA-seq for transcriptomics research.Competing Interest StatementThe authors have declared no competing interest.},
        URL = {https://www.biorxiv.org/content/early/2024/10/25/2024.10.23.619929},
        eprint = {https://www.biorxiv.org/content/early/2024/10/25/2024.10.23.619929.full.pdf},
        journal = {bioRxiv}
}
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

- 📖 Check our [Documentation](./documentation/tutorial.md)
- 🐛 [Report issues](https://github.com/ylab-hi/DeepChopper/issues)

---

DeepChopper is developed with ❤️ by the YLab team.
Happy sequencing! 🧬🔬

[pypi]: https://pypi.python.org/pypi/deepchopper
