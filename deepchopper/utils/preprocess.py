from pathlib import Path


def load_kmer2id(kmer2id_file: Path) -> dict:
    """Load a dictionary mapping kmer strings to integer IDs from a file.

    Parameters:
    kmer2id_file (Path): A Path object pointing to the file containing the kmer to ID mapping.

    Returns:
    dict: A dictionary mapping kmer strings to integer IDs.

    Raises:
    FileNotFoundError: If the specified file does not exist.
    ValueError: If the file format is incorrect or if the kmer and ID cannot be properly parsed.

    Example:
    If the kmer2id_file contains the following lines:
    ATG 1
    CAG 2
    The function will return {'ATG': 1, 'CAG': 2}.
    """
    kmer2id_file = Path(kmer2id_file)

    kmer2id = {}
    with Path(kmer2id_file).open("r") as f:
        for line in f:
            kmer, idx = line.strip().split()
            kmer2id[kmer] = int(idx)
    return kmer2id
