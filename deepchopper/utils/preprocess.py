from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file, save_file


def load_safetensor(file_path: Path, device="cpu") -> dict[str, torch.Tensor]:
    file_path = Path(file_path)
    return load_file(file_path, device=device)


def save_ndarray_to_safetensor(ndarray: dict[str, np.ndarray], file_path: Path):
    """Save a dictionary of NumPy arrays as PyTorch tensors to a file.

    Parameters:
    - ndarray (dict[str, np.ndarray]): A dictionary where keys are strings and values are NumPy arrays.
    - file_path (Path): The file path where the PyTorch tensors will be saved.

    Returns:
    - None

    Raises:
    - ValueError: If the input dictionary is empty.
    - TypeError: If the input dictionary values are not NumPy arrays.
    - FileNotFoundError: If the specified file path does not exist.
    - OSError: If there is an issue with writing to the file.

    This function converts each NumPy array in the input dictionary to a PyTorch tensor and saves the resulting dictionary
    to a file specified by the file path. If any errors occur during the process, appropriate exceptions are raised.
    """
    tensor_dict = {key: torch.tensor(value) for key, value in ndarray.items()}
    save_file(tensor_dict, file_path)


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
