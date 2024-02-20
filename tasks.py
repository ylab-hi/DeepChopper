import logging
from pathlib import Path

import numpy as np
from invoke import task
from rich.logging import RichHandler

import deepchopper


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
    Path("data.npz").unlink(missing_ok=True)
    Path("kmer2idx.txt").unlink(missing_ok=True)


@task
def encode_parqut(c, file: Path, level="info"):
    from deepchopper import encode_fq_path_to_parquet

    if level == "info":
        level = logging.INFO
    elif level == "debug":
        level = logging.DEBUG
    elif level == "warn":
        level = logging.WARNING
    else:
        level = logging.INFO

    FORMAT = "%(message)s"
    logging.basicConfig(
        level=level,
        format=FORMAT,
        handlers=[RichHandler()],
    )
    data = Path(file)
    k = 3
    bases = "ACGTN"
    qual_offset = 33

    encode_fq_path_to_parquet(data, k, bases, qual_offset, vectorized_target=False)
    # import pyarrow.parquet as pq
    # df = pq.read_table(result_path)
    # df_pd = df.to_pandas()
    # assert df_pd.shape == (25, 5)


@task
def encode_tensor(c, file: Path, level="info"):
    clean(c)

    if level == "info":
        level = logging.INFO
    elif level == "debug":
        level = logging.DEBUG
    elif level == "warn":
        level = logging.WARNING
    else:
        level = logging.INFO

    FORMAT = "%(message)s"
    logging.basicConfig(
        level=level,
        format=FORMAT,
        handlers=[RichHandler()],
    )

    data = Path(file)
    k = 3
    bases = "ACGTN"
    qual_offset = 33

    inputs, target, qual, kmer2idx = encode_fq_path(data, k, bases, qual_offset, True)

    logging.info(f"inputs.shape: {inputs.shape}")
    logging.info(f"target.shape: {target.shape}")
    logging.info(f"qual.shape: {qual.shape}")
    logging.info(f"len(kmer2idx): {len(kmer2idx)}")

    np.savez("data.npz", inputs=inputs, target=target, qual=qual)

    with open("kmer2idx.txt", "w") as f:
        for kmer, idx in kmer2idx.items():
            f.write(f"{kmer}\t{idx}\n")


@task
def encode_fqs(c, data_folder):
    from deepchopper.data import encode_fq_files_in_folder_to_parquet

    data_folder = Path(data_folder)
    encode_fq_files_in_folder_to_parquet(data_folder)


@task
def convert_safe(c, file):
    from deepchopper.utils import save_ndarray_to_safetensor

    file = Path(file)
    data = np.load(file)

    save_ndarray_to_safetensor(data, file.with_suffix(".safetensors"))


@task
def readq(c, file):
    import pyarrow.parquet as pq

    df = pq.read_table(file)

    df_pd = df.to_pandas()
    print(df_pd.shape)
    print(df_pd.head())


    from pyarrow import json
    import json
    df_dict = df.to_pydict()
    json.dump(df_dict, open("df.json", "w"))
