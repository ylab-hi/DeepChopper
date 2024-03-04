import multiprocessing
from pathlib import Path

from datasets import load_dataset


def collect_and_split_dataset(
    internal_fq_path: str | Path,
    terminal_fq_path: str | Path,
    negative_fq_path: str | Path,
    total_reads: float,
    train_ratio: float,  # 0.8
    val_ratio: float,  # 0.1
    test_ratio: float,  # 0.1
    iternal_adapter_ratio: float,
    positive_ratio: float,
):
    internal_fq_path = Path(internal_fq_path)
    terminal_fq_path = Path(terminal_fq_path)
    negative_fq_path = Path(negative_fq_path)

    if (
        not internal_fq_path.exists()
        or not terminal_fq_path.exists()
        or not negative_fq_path.exists()
    ):
        raise FileNotFoundError("One or more of the input files does not exist.")

    if (
        (internal_fq_path.suffix != ".parquet")
        or (terminal_fq_path.suffix != ".parquet")
        or (negative_fq_path.suffix != ".parquet")
    ):
        raise ValueError("Input files must be in .parquet format.")

    if train_ratio + val_ratio + test_ratio != 1.0:
        message = "train_ratio + val_ratio + test_ratio must be equal to 1.0"
        raise ValueError(message)

    terminal_adapter_ratio = 1.0 - iternal_adapter_ratio
    negative_ratio = 1.0 - positive_ratio

    # calculate the number of reads in each file
    train_count = total_reads * train_ratio
    val_count = total_reads * val_ratio
    test_count = total_reads * test_ratio

    train_positive_count = train_count * positive_ratio
    val_positive_count = val_count * positive_ratio
    test_positive_count = test_count * positive_ratio

    train_count * negative_ratio
    val_count * negative_ratio
    test_count * negative_ratio

    train_positive_count * iternal_adapter_ratio
    train_positive_count * terminal_adapter_ratio

    val_positive_count * iternal_adapter_ratio
    val_positive_count * terminal_adapter_ratio

    test_positive_count * iternal_adapter_ratio
    test_positive_count * terminal_adapter_ratio


def load_and_split_dataset(data_file: str | Path, num_proc: int | None = None):
    """Load and split a dataset into training, validation, and testing sets.

    Args:
        data_file (str, Path): A dictionary containing the file paths for the dataset.
        num_proc (int, optional): The number of processes to use for loading the dataset. Defaults to None.

    Returns:
        Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset, torch.utils.data.Dataset]: A tuple containing the training, validation, and testing datasets.

    Example:
        data_files = {"train": "train.parquet"}
        train_dataset, val_dataset, test_dataset = load_and_split_dataset(data_files)
    """
    if num_proc is None:
        num_proc = multiprocessing.cpu_count()

    data_files = {"train": str(data_file)}
    train_dataset = load_dataset(
        "parquet",
        data_files=data_files,
        num_proc=num_proc,
        split="train[:80%]",
    ).with_format("torch")

    val_dataset = load_dataset(
        "parquet", data_files=data_files, num_proc=num_proc, split="train[80%:90%]"
    ).with_format("torch")

    test_dataset = load_dataset(
        "parquet", data_files=data_files, num_proc=num_proc, split="train[90%:]"
    ).with_format("torch")
    return train_dataset, val_dataset, test_dataset
