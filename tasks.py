import logging
from pathlib import Path

import numpy as np
from invoke import task
from rich.logging import RichHandler

import deepchopper
from deepchopper import encode_fq_path


@task
def log(c, level="info"):
    FORMAT = "%(message)s"

    if level == "info":
        level = logging.INFO
    elif level == "debug":
        level = logging.DEBUG
    elif level == "warning":
        level = logging.WARNING
    else:
        level = logging.INFO

    logging.basicConfig(
        level=level,
        format=FORMAT,
        handlers=[RichHandler()],
    )
    deepchopper.test_log()


@task
def clean(c):
    Path("inputs.npz").unlink(missing_ok=True)
    Path("target.npz").unlink(missing_ok=True)
    Path("qual.npz").unlink(missing_ok=True)
    Path("kmer2idx.txt").unlink(missing_ok=True)


@task
def encode(c, file: Path, level="info"):
    clean(c)

    if level == "info":
        level = logging.INFO
    elif level == "debug":
        level = logging.DEBUG
    elif level == "warning":
        level = logging.WARNING
    else:
        level = logging.INFO

    FORMAT = "%(message)s"
    logging.basicConfig(
        level=level,
        format=FORMAT,
        handlers=[RichHandler()],
    )

    data = file
    k = 3
    bases = "ACGTN"
    qual_offset = 33
    inputs, target, qual, kmer2idx = encode_fq_path(data, k, bases, qual_offset, True)

    np.savez("inputs.npz", inputs)
    np.savez("target.npz", target)
    np.savez("qual.npz", qual)

    with open("kmer2idx.txt", "w") as f:
        for kmer, idx in kmer2idx.items():
            f.write(f"{kmer}\t{idx}\n")
