import multiprocessing
from pathlib import Path

from datasets import load_dataset


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

    val_dataset = load_dataset("parquet", data_files=data_files, num_proc=num_proc, split="train[80%:90%]").with_format(
        "torch"
    )

    test_dataset = load_dataset("parquet", data_files=data_files, num_proc=num_proc, split="train[90%:]").with_format(
        "torch"
    )
    return train_dataset, val_dataset, test_dataset
