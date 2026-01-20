# **DeepChopper** [![social](https://img.shields.io/github/stars/ylab-hi/DeepChopper?style=social)](https://github.com/ylab-hi/DeepChopper/stargazers)

[![pypi](https://img.shields.io/pypi/v/deepchopper.svg)](https://pypi.python.org/pypi/deepchopper)
[![PyPI - Wheel](https://img.shields.io/pypi/wheel/deepchopper)](https://pypi.org/project/deepchopper/#files)
[![license](https://img.shields.io/pypi/l/deepchopper.svg)](https://github.com/ylab-hi/DeepChopper/blob/main/LICENSE)
[![pypi version](https://img.shields.io/pypi/pyversions/deepchopper.svg)](https://pypi.python.org/pypi/deepbiop)
[![platform](https://img.shields.io/badge/platform-linux%20%7C%20osx%20%7C%20win-blue)](https://pypi.org/project/deepchopper/#files)
[![Actions status](https://github.com/ylab-hi/DeepChopper/actions/workflows/release-python.yml/badge.svg)](https://github.com/ylab-hi/DeepChopper/actions)
[![Space](https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-hf-spaces-md.svg)](https://huggingface.co/spaces/yangliz5/deepchopper)

## Overview

🧬 **DeepChopper** is a genomic language model designed to accurately detect and remove chimeric artifacts in Nanopore direct RNA sequencing (dRNA-seq) data. By leveraging deep learning, DeepChopper identifies adapter sequences within base-called reads, ensuring higher quality and more reliable sequencing results.

---

## :star: Key Features

<div class="grid cards" markdown>

- :material-target:{ .feature-icon } **High Accuracy**

    ---

    State-of-the-art detection of chimeric reads using transformer-based models with >95% sensitivity

- :material-lightning-bolt:{ .feature-icon } **Fast Processing**

    ---

    Optimized Rust core with parallel processing capabilities. Process millions of reads in minutes

- :material-sync:{ .feature-icon } **Zero-shot Capability**

    ---

    Works across different RNA chemistries (RNA002, RNA004, and newer) without retraining

- :material-language-python:{ .feature-icon } **Easy Integration**

    ---

    Simple Python API and CLI for seamless workflow integration

- :material-chip:{ .feature-icon } **GPU Acceleration**

    ---

    Optional GPU support (NVIDIA, Apple Silicon) for faster processing of large datasets

- :material-web:{ .feature-icon } **Web Interface**

    ---

    Interactive web UI for quick testing and visualization

</div>

---

## :material-help-circle: Why DeepChopper?

!!! info "The Problem"
    Chimera artifacts in nanopore dRNA-seq can confound transcriptome analyses, leading to **false gene fusion calls** and **incorrect transcript annotations**. Existing basecalling tools fail to detect internal adapter sequences, leaving these artifacts in your data.

!!! success "The Solution"
    DeepChopper solves this problem with a three-step approach:

    1. :material-magnify: **Detecting** adapter sequences that basecallers miss
    2. :material-content-cut: **Chopping** reads at adapter locations to remove chimeric artifacts  
    3. :material-shield-check: **Preserving** high-quality sequence data for downstream analysis

## Quick Start

### Try Online

Experience DeepChopper instantly without any installation:

[![Open in Hugging Face Spaces](https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-hf-spaces-md.svg)](https://huggingface.co/spaces/yangliz5/deepchopper)

!!! note
    The online version is limited to one FASTQ record at a time. For large-scale analyses, please install DeepChopper locally.

### Installation

Install DeepChopper using pip:

```bash
pip install deepchopper
```

Verify the installation:

```bash
deepchopper --help
```

For detailed installation instructions, see the [Installation Guide](installation.md).

### Basic Usage

```bash
# 1. Predict chimera artifacts (automatically encodes FASTQ data)
deepchopper predict raw_reads.fastq --output predictions

# 2. Chop the reads at detected adapter locations
deepchopper chop predictions/ raw_reads.fastq --output chopped.fastq
```

For a complete walkthrough, check out the [Tutorial](tutorial.md).

---

## :material-application: Use Cases

<div class="grid cards" markdown>

- :material-dna:{ .feature-icon } **Transcriptome Assembly**

    ---

    Remove chimera artifacts to improve transcript reconstruction and assembly quality

- :material-link-variant:{ .feature-icon } **Gene Fusion Detection**

    ---

    Eliminate false positives from adapter-bridged artifacts for accurate fusion calling

- :material-chart-bar:{ .feature-icon } **Differential Expression**

    ---

    Ensure accurate read counts by removing chimeric reads before quantification

- :material-quality-high:{ .feature-icon } **RNA-Seq QC**

    ---

    Assess and improve data quality in dRNA-seq experiments

</div>

---

## Citation

If DeepChopper helps your research, please cite our paper:

```bibtex
@article{li2026genomic,
  title = {Genomic Language Model Mitigates Chimera Artifacts in Nanopore Direct {{RNA}} Sequencing},
  author = {Li, Yangyang and Wang, Ting-You and Guo, Qingxiang and Ren, Yanan and Lu, Xiaotong and Cao, Qi and Yang, Rendong},
  date = {2026-01-19},
  journaltitle = {Nature Communications},
  shortjournal = {Nat Commun},
  publisher = {Nature Publishing Group},
  issn = {2041-1723},
  doi = {10.1038/s41467-026-68571-5},
  url = {https://www.nature.com/articles/s41467-026-68571-5},
  urldate = {2026-01-20}
}
```

## Support

- 📖 [Documentation](tutorial.md)
- 🐛 [Issue Tracker](https://github.com/ylab-hi/DeepChopper/issues)
- 💬 [GitHub Discussions](https://github.com/ylab-hi/DeepChopper/discussions)

## License

DeepChopper is released under the [Apache License 2.0](https://github.com/ylab-hi/DeepChopper/blob/main/LICENSE).

---

<div align="center">
  Developed with ❤️ by the YLab team | Happy sequencing! 🧬🔬
</div>
