import logging
from pathlib import Path

from rich.logging import RichHandler

from deepchopper.deepchopper import encode_fq_path_to_parquet


def encode_one_fq_file(
    fq_file: Path,
    kmer_size: int = 3,  # unused for encode parquet
    qual_offset: int = 33,
    bases="ACGTN",
):
    """Encode the sequences in a single FASTQ file into numerical representations and save the encoded data."""
    encode_fq_path_to_parquet(fq_file, kmer_size, bases, qual_offset, vectorized_target=False)


def encode_fq_files_in_folder_to_parquet(data_folder: Path):
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

    for fq_file in data_folder.glob("*.[fq|fastq]"):
        logging.info(f"Encoding {fq_file}")
        encode_one_fq_file(fq_file)
