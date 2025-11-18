import multiprocessing
from collections.abc import Iterator
from functools import partial
from pathlib import Path
from typing import Any

import pyfastx
from datasets import Dataset as HuggingFaceDataset
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

import deepchopper
from deepchopper.models.llm import (
    DataCollatorForTokenClassificationWithQual,
    tokenize_and_align_labels_and_quals,
    tokenize_and_align_labels_and_quals_ids,
)


def parse_fastq_file(file_path: Path, has_targets: bool = True) -> Iterator[dict]:
    """Parse a FastQ file using pyfastx and return a dictionary.

    Args:
        file_path: Path to the FastQ file (.fq, .fastq, .fq.gz, .fastq.gz)
        has_targets: Whether the file contains target labels in the identifier line

    Raises:
        ValueError: If file is empty, corrupted, or contains invalid records
        RuntimeError: If parsing fails
    """
    try:
        # Use pyfastx to parse the file
        fq = pyfastx.Fastx(str(file_path), uppercase=True)

        record_count = 0
        for name, seq, qual in fq:
            # Validate record completeness
            if not name or not seq or not qual:
                msg = f"Incomplete FASTQ record at position {record_count} in {file_path}"
                raise ValueError(msg)

            # Validate sequence and quality lengths match
            if len(seq) != len(qual):
                msg = f"Sequence/quality length mismatch in record '{name}': seq={len(seq)}, qual={len(qual)}"
                raise ValueError(msg)

            # Parse target if present
            target = [0, 0]
            if has_targets:
                try:
                    target = deepchopper.parse_target_from_id(name)
                except Exception as e:
                    msg = f"Failed to parse target from ID '{name}': {e}"
                    raise ValueError(msg) from e

            encoded_qual = deepchopper.encode_qual(qual, deepchopper.default.QUAL_OFFSET)

            yield {
                "id": name,
                "seq": seq,
                "qual": encoded_qual,
                "target": target,
            }

            record_count += 1

        # Ensure we read at least one record
        if record_count == 0:
            msg = f"No valid records found in {file_path}"
            raise ValueError(msg)

    except pyfastx.FastxError as e:
        msg = f"FASTQ parsing error in {file_path}: {e}"
        raise RuntimeError(msg) from e
    except Exception as e:
        # Re-raise ValueError and RuntimeError as-is
        if isinstance(e, (ValueError, RuntimeError)):
            raise
        msg = f"Error parsing FastQ file {file_path}: {e}"
        raise RuntimeError(msg) from e


class OnlyFqDataModule(LightningDataModule):
    """PyTorch Lightning DataModule for genomic sequence data in FASTQ format.

    This DataModule is designed to handle FASTQ files containing DNA or RNA sequences,
    along with their associated quality scores and optional target labels embedded in the sequence identifiers.
    It parses FASTQ files using pyfastx, encodes quality scores, and extracts targets for supervised learning tasks.

    The module provides train, validation, test, and predict dataloaders compatible with PyTorch Lightning workflows.
    It supports integration with HuggingFace tokenizers and custom data collators for token classification tasks.

    Expected input:
        - FASTQ files (.fq, .fastq, .fq.gz, .fastq.gz) with sequence identifiers optionally containing target labels.
        - Each record includes a sequence, quality string, and (optionally) a target label.

    Key features:
        - Efficient parsing of large FASTQ files using pyfastx.
        - Encoding of quality scores for model input.
        - Extraction of target labels from sequence identifiers.
        - Customizable data collation and tokenization for downstream models.

    Implements the standard LightningDataModule interface:
        - prepare_data
        - setup
        - train_dataloader
        - val_dataloader
        - test_dataloader
        - predict_dataloader
        - teardown

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
        predict_data_path: Path | None = None,
        batch_size: int = 12,
        num_workers: int = 0,
        max_train_samples: int | None = None,
        max_val_samples: int | None = None,
        max_test_samples: int | None = None,
        max_predict_samples: int | None = None,
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

        if self.hparams.predict_data_path is not None:
            data_paths.append(self.hparams.predict_data_path)
            # no need to prepare data for prediction
            return

        for data_path in data_paths:
            if not Path(data_path).exists():
                msg = f"Data file {data_path} does not exist."
                raise ValueError(msg)

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

        if stage == "predict":
            if not self.hparams.predict_data_path:
                msg = "Predict data path is required for prediction stage."
                raise ValueError(msg)

            # Calculate appropriate num_proc based on file size
            file_path = Path(self.hparams.predict_data_path)
            file_size_gb = file_path.stat().st_size / (1024**3)

            base_num_proc = min(self.hparams.num_workers, multiprocessing.cpu_count() - 1)

            # Reduce parallelism for large files to avoid memory issues
            if file_size_gb > 1.0:  # Files larger than 1GB
                num_proc = min(max(1, base_num_proc), 4)
                import logging

                logging.info(f"Large file detected ({file_size_gb:.2f}GB), limiting num_proc to {num_proc}")
            else:
                num_proc = max(1, base_num_proc)

            predict_dataset = HuggingFaceDataset.from_generator(
                parse_fastq_file,
                gen_kwargs={"file_path": self.hparams.predict_data_path, "has_targets": False},
                num_proc=num_proc,
            ).with_format("torch")

            self.data_predict = predict_dataset.map(
                partial(
                    tokenize_and_align_labels_and_quals_ids,
                    tokenizer=self.hparams.tokenizer,
                    max_length=self.hparams.tokenizer.max_len_single_sentence,
                ),
                num_proc=max(1, num_proc),  # type: ignore
            ).remove_columns(["seq", "qual", "target"])
            del predict_dataset
            return

        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            num_proc = min(self.hparams.num_workers, multiprocessing.cpu_count() - 1)
            data_files = {}
            data_files["train"] = self.hparams.train_data_path

            if self.hparams.val_data_path is None:
                msg = "Please provide a validation data path."
                raise ValueError(msg)

            if self.hparams.test_data_path is None:
                msg = "Please provide a test data path."
                raise ValueError(msg)

            train_dataset = HuggingFaceDataset.from_generator(
                parse_fastq_file,
                gen_kwargs={"file_path": self.hparams.train_data_path, "has_targets": True},
                num_proc=max(1, num_proc),
            ).with_format("torch")

            val_dataset = HuggingFaceDataset.from_generator(
                parse_fastq_file,
                gen_kwargs={"file_path": self.hparams.val_data_path, "has_targets": True},
                num_proc=max(1, num_proc),
            ).with_format("torch")

            test_dataset = HuggingFaceDataset.from_generator(
                parse_fastq_file,
                gen_kwargs={"file_path": self.hparams.test_data_path, "has_targets": True},
                num_proc=max(1, num_proc),
            ).with_format("torch")

            if self.hparams.max_train_samples is not None:
                max_train_samples = min(self.hparams.max_train_samples, len(train_dataset))
                train_dataset = train_dataset.select(range(max_train_samples))

            if self.hparams.max_val_samples is not None:
                max_val_samples = min(self.hparams.max_val_samples, len(val_dataset))
                val_dataset = val_dataset.select(range(max_val_samples))

            if self.hparams.max_test_samples is not None:
                max_test_samples = min(self.hparams.max_test_samples, len(test_dataset))
                test_dataset = test_dataset.select(range(max_test_samples))

            self.data_train = train_dataset.map(
                partial(
                    tokenize_and_align_labels_and_quals,
                    tokenizer=self.hparams.tokenizer,
                    max_length=self.hparams.tokenizer.max_len_single_sentence,
                ),
                num_proc=max(1, num_proc),  # type: ignore
            ).remove_columns(["id", "seq", "qual", "target"])

            self.data_val = val_dataset.map(
                partial(
                    tokenize_and_align_labels_and_quals,
                    tokenizer=self.hparams.tokenizer,
                    max_length=self.hparams.tokenizer.max_len_single_sentence,
                ),
                num_proc=max(1, num_proc),  # type: ignore
            ).remove_columns(["id", "seq", "qual", "target"])

            self.data_test = test_dataset.map(
                partial(
                    tokenize_and_align_labels_and_quals,
                    tokenizer=self.hparams.tokenizer,
                    max_length=self.hparams.tokenizer.max_len_single_sentence,
                ),
                num_proc=max(1, num_proc),  # type: ignore
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

    def predict_dataloader(self) -> DataLoader[Any]:
        """Create and return the predict dataloader.

        :return: The predict dataloader.
        """
        return DataLoader(
            dataset=self.data_predict,
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
