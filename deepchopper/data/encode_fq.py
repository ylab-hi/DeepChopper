import logging
from pathlib import Path

import numpy as np
from rich.logging import RichHandler

from ..deepchopper import encode_fq_path  # noqa:TID252


def encode_one_fq_file(
    fq_file: Path,
    kmer_size: int = 3,
    qual_offset: int = 33,
    bases="ACGTN",
):
    """Encode the sequences in a single FASTQ file into numerical representations and save the encoded data.

    Parameters:
    - fq_file (Path): Path to the input FASTQ file to be encoded.
    - kmer_size (int): Size of the k-mer to use for encoding. Default is 3.
    - qual_offset (int): Quality score offset to use for encoding. Default is 33.

    Returns:
    - None

    This function encodes the sequences in the input FASTQ file using the specified k-mer size and quality score offset.
    It then saves the encoded data as numpy arrays and a k-mer to index mapping file.

    The encoded data includes:
    - inputs: Encoded sequences as numpy array.
    - target: Encoded target sequences as numpy array.
    - qual: Encoded quality scores as numpy array.
    - kmer2idx: Mapping of k-mers to their corresponding indices.

    The encoded data is saved in a .npz file with the format '{fq_name}_data.npz'.
    The k-mer to index mapping is saved in a .txt file with the format '{fq_name}_kmer2idx.txt'.

    Example usage:
    encode_one_fq_file(Path('input.fq'), kmer_size=4, qual_offset
    """
    fq_name = fq_file.stem
    fq_folder = fq_file.parent
    inputs, target, qual, kmer2idx = encode_fq_path(fq_file, kmer_size, bases, qual_offset, vectorized_target=True)

    logging.info(f"inputs.shape: {inputs.shape}")
    logging.info(f"target.shape: {target.shape}")
    logging.info(f"qual.shape: {qual.shape}")
    logging.info(f"len(kmer2idx): {len(kmer2idx)}")

    np.savez(fq_folder / f"{fq_name}_data.npz", inputs=inputs, target=target, qual=qual)
    with Path(fq_folder / f"{fq_name}_kmer2idx.txt").open("w") as f:
        for kmer, idx in kmer2idx.items():
            f.write(f"{kmer}\t{idx}\n")


def encode_fq_files_in_folder(data_folder: Path):
    """Encode all fastq files in a given folder.

    Args:
        data_folder (Path): The folder containing the fastq files to encode.

    Raises:
        FileNotFoundError: If the specified data_folder does not exist.
    """
    FORMAT = "%(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=FORMAT,
        handlers=[RichHandler()],
    )

    if not data_folder.exists():
        msg = f"Folder {data_folder} does not exist."
        logging.error(msg)

    for fq_file in data_folder.glob("*.fq"):
        logging.info(f"Encoding {fq_file}")
        encode_one_fq_file(fq_file)
