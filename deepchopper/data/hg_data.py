import multiprocessing

from datasets import load_dataset


def load_and_split_dataset(data_files: dict[str, str], num_proc: int | None = None):
    if num_proc is None:
        num_proc = multiprocessing.cpu_count()

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
