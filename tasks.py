import logging
from pathlib import Path

import numpy as np
from invoke import task
from rich.logging import RichHandler

from deepchopper import encode_fq_path


@task
def clean(c):
    Path("inputs.npz").unlink(missing_ok=True)
    Path("target.npz").unlink(missing_ok=True)
    Path("qual.npz").unlink(missing_ok=True)
    Path("kmer2idx.txt").unlink(missing_ok=True)


@task
def encode(c, file: Path):
    clean(c)
    FORMAT = "%(message)s"
    logging.basicConfig(
        level=logging.DEBUG,
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
