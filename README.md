# <img src="./documentation/logo.webp" alt="logo" height=100> **DeepChopper** [![social](https://img.shields.io/github/stars/ylab-hi/DeepChopper?style=social)](https://github.com/ylab-hi/DeepChopper/stargazers)

<!--toc:start-->

- [Feature](#feature)
- [🤜 Contribution](#%F0%9F%A4%9C-contribution)

<!--toc:end-->

Language models identify chimeric artificial reads in NanoPore direct-RNA sequencing data.
DeepChopper leverages language model to accurately detect and chop these aritificial sequences which may cause chimeric reads, ensuring higher quality and more reliable sequencing results.
By integrating seamlessly with existing workflows, DeepChopper provides a robust solution for researchers and bioinformaticians working with NanoPorea direct-RNA sequencing data.

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
   deepchopper --version
   ```

For GPU support, ensure you have CUDA installed on your system, then install the GPU version:

```bash
pip install deepchopper[gpu]
```

Note: If you encounter any issues,
please check our GitHub repository for troubleshooting guides or to report a problem.

## Usage

```bash
deepchopper endcode --input <input>
```

```bash
deepchopper predict --input <input> --output <output>
```

```bash
deepchopper chop --input <input> --output <output>
```

## Cite

## 🤜 Contribution

**Build Environment**

```bash
git clone https://github.com/ylab-hi/DeepChopper.git
cd DeepChopper
conda env create -n environment.yaml
conda activate deepchopper
```

**Install Dependencies**

```bash
pip install pipx
pipx install --suffix @master git+https://github.com/python-poetry/poetry.git@master
poetry@master install
```
