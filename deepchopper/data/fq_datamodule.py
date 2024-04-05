from __future__ import annotations

import multiprocessing
from functools import partial
from typing import TYPE_CHECKING, Any

from datasets import load_dataset
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from deepchopper.deepchopper import (
    default,
    encode_fq_path_to_parquet,
)
from deepchopper.models.hyena import (
    DataCollatorForTokenClassificationWithQual,
    tokenize_and_align_labels_and_quals,
)

if TYPE_CHECKING:
    from pathlib import Path

    from transformers import AutoTokenizer


class FqDataModule(LightningDataModule):
    """`LightningDataModule` for the MNIST dataset.

    The MNIST database of handwritten digits has a training set of 60,000 examples, and a test set of 10,000 examples.
    It is a subset of a larger set available from NIST. The digits have been size-normalized and centered in a
    fixed-size image. The original black and white images from NIST were size normalized to fit in a 20x20 pixel box
    while preserving their aspect ratio. The resulting images contain grey levels as a result of the anti-aliasing
    technique used by the normalization algorithm. the images were centered in a 28x28 image by computing the center of
    mass of the pixels, and translating the image so as to position this point at the center of the 28x28 field.

    A `LightningDataModule` implements 7 key methods:

    ```python
        def prepare_data(self):
        # Things to do on 1 GPU/TPU (not on every GPU/TPU in DDP).
        # Download data, pre-process, split, save to disk, etc...

        def setup(self, stage):
        # Things to do on every process in DDP.
        # Load data, set variables, etc...

        def train_dataloader(self):
        # return train dataloader

        def val_dataloader(self):
        # return validation dataloader

        def test_dataloader(self):
        # return test dataloader

        def predict_dataloader(self):
        # return predict dataloader

        def teardown(self, stage):
        # Called on every process in DDP.
        # Clean up after fit or test.
    ```

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://lightning.ai/docs/pytorch/latest/data/datamodule.html
    """

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        train_data_path: Path,
        val_data_path: Path | None = None,
        test_data_path: Path | None = None,
        train_val_test_split: tuple[float, float, float] = (0.8, 0.1, 0.1),
        batch_size: int = 12,
        num_workers: int = 0,
        *,
        pin_memory: bool = False,
    ) -> None:
        """Initialize a `FqDataModule`.

        :param train_val_test_split: The train, validation and test split. Defaults to `(55_000, 5_000, 10_000)`.
        :param batch_size: The batch size. Defaults to `64`.
        :param num_workers: The number of workers. Defaults to `0`.
        :param pin_memory: Whether to pin memory. Defaults to `False`.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.data_train: Dataset | None = None
        self.data_val: Dataset | None = None
        self.data_test: Dataset | None = None
        self.batch_size_per_device = batch_size
        self.data_collator = DataCollatorForTokenClassificationWithQual(tokenizer)

    @property
    def num_classes(self) -> int:
        """Get the number of classes.

        :return: The number of MNIST classes (10).
        """
        return 2

    def prepare_data(self) -> None:
        """Encode the FastQ data to Parquet format."""
        data_paths = [self.hparams.train_data_path]

        if self.hparams.val_data_path is not None:
            data_paths.append(self.hparams.val_data_path)

        if self.hparams.test_data_path is not None:
            data_paths.append(self.hparams.test_data_path)

        for data_path in data_paths:
            if data_path in (".fq", ".fastq"):
                encode_fq_path_to_parquet(
                    data_path,
                    default.KMER_SIZE,
                    bases=default.BASES,
                    qual_offset=default.QUAL_OFFSET,
                    vectorized_target=default.VECTORIZED_TARGET,
                )
            elif data_path.suffix == ".parquet":
                pass
            else:
                msg = f"Data file {data_path} is not in FastQ or Parquet format."
                raise ValueError(msg)

        self.hparams.train_data_path = self.hparams.train_data_path.with_suffix(".parquet")

        if self.hparams.val_data_path is not None:
            self.hparams.val_data_path = self.hparams.val_data_path.with_suffix(".parquet")

        if self.hparams.test_data_path is not None:
            self.hparams.test_data_path = self.hparams.test_data_path.with_suffix(".parquet")

    def setup(self, stage: str | None = None) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by Lightning before `trainer.fit()`, `trainer.validate()`, `trainer.test()`, and
        `trainer.predict()`, so be careful not to execute things like random split twice! Also, it is called after
        `self.prepare_data()` and there is a barrier in between which ensures that all the processes proceed to
        `self.setup()` once the data is prepared and available for use.

        :param stage: The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`. Defaults to ``None``.
        """
        # Divide batch size by the number of devices.
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                msg = f"Batch size ({self.hparams.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                raise RuntimeError(msg)
            self.batch_size_per_device = self.hparams.batch_size // self.trainer.world_size

        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            num_proc = multiprocessing.cpu_count()
            data_files = {}
            data_files["train"] = self.hparams.train_data_path

            if self.hparams.val_data_path is not None:
                data_files["validation"] = self.hparams.val_data_path

            if self.hparams.test_data_path is not None:
                data_files["test"] = self.hparams.test_data_path

            if self.hparams.val_data_path is None or self.hparams.test_data_path is None:
                split_percent = self.hparams.train_val_test_split * 100

                train_dataset = load_dataset(
                    "parquet",
                    data_files=data_files,
                    num_proc=num_proc,
                    split=f"train[:{split_percent[0]}%]",
                ).with_format("torch")

                val_dataset = load_dataset(
                    "parquet",
                    data_files=data_files,
                    num_proc=num_proc,
                    split=f"train[{split_percent[0]}%:{split_percent[0] + split_percent[1]}%]",
                ).with_format("torch")

                test_dataset = load_dataset(
                    "parquet",
                    data_files=data_files,
                    num_proc=num_proc,
                    split=f"train[{split_percent[0] + split_percent[1]}%:]",
                ).with_format("torch")

            else:
                raw_datasets = load_dataset(
                    "parquet", data_files=data_files, num_proc=num_proc
                ).with_format("torch")

                train_dataset = raw_datasets["train"]
                val_dataset = raw_datasets["validation"]
                test_dataset = raw_datasets["test"]

            self.data_train = train_dataset.map(
                partial(
                    tokenize_and_align_labels_and_quals,
                    tokenizer=self.hparams.tokenizer,
                    max_length=self.hparams.tokenizer.max_len_single_sentence,
                ),
                num_proc=multiprocessing.cpu_count(),  # type: ignore
            ).remove_columns(["id", "seq", "qual", "target"])

            self.data_val = val_dataset.map(
                partial(
                    tokenize_and_align_labels_and_quals,
                    tokenizer=self.hparams.tokenizer,
                    max_length=self.hparams.tokenizer.max_len_single_sentence,
                ),
                num_proc=multiprocessing.cpu_count(),  # type: ignore
            ).remove_columns(["id", "seq", "qual", "target"])

            self.data_test = test_dataset.map(
                partial(
                    tokenize_and_align_labels_and_quals,
                    tokenizer=self.hparams.tokenizer,
                    max_length=self.hparams.tokenizer.max_len_single_sentence,
                ),
                num_proc=multiprocessing.cpu_count(),  # type: ignore
            ).remove_columns(["id", "seq", "qual", "target"])

            del train_dataset, val_dataset, test_dataset

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=self.data_collator.torch_call,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=self.data_collator.torch_call,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader.

        :return: The test dataloader.
        """
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=self.data_collator.torch_call,
            shuffle=False,
        )

    def teardown(self, stage: str | None = None) -> None:
        """Lightning hook for cleaning up after `trainer.fit()`, `trainer.validate()`,.

        `trainer.test()`, and `trainer.predict()`.

        :param stage: The stage being torn down. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
            Defaults to ``None``.
        """

    def state_dict(self) -> dict[Any, Any]:
        """Called when saving a checkpoint. Implement to generate and save the datamodule state.

        :return: A dictionary containing the datamodule state that you want to save.
        """
        return {}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Called when loading a checkpoint. Implement to reload datamodule state given datamodule.

        `state_dict()`.

        :param state_dict: The datamodule state returned by `self.state_dict()`.
        """
