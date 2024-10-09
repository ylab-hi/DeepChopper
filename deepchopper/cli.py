import logging
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import lightning
import torch
import typer
from rich import print
from rich.logging import RichHandler

import deepchopper

from .deepchopper import (
    default,
    encode_fq_path_to_parquet,
    encode_fq_path_to_parquet_chunk,
)
from .utils import (
    highlight_target,
)

if TYPE_CHECKING:
    from lightning.pytorch import LightningDataModule

FORMAT = "%(message)s"
logging.basicConfig(
    level=logging.INFO,
    format=FORMAT,
    handlers=[RichHandler()],
)


def random_show_seq(dataset, sample: int = 3):
    """Randomly selects 'sample' number of sequences from the given dataset and prints their IDs and targets.

    Parameters:
        dataset : A list of dictionaries where each dictionary represents a sequence with keys 'id', 'seq', and 'target'.
        sample (int): The number of sequences to randomly select from the dataset. Default is 3.
    """
    total = len(dataset)
    import secrets

    highlight_ids = (secrets.randbelow(total) for _ in range(sample))
    for highlight_id in highlight_ids:
        print(f"id: {dataset[highlight_id]['id']}")
        highlight_target(dataset[highlight_id]["seq"], *dataset[highlight_id]["target"])


app = typer.Typer(
    context_settings={"help_option_names": ["-h", "--help"]},
)


@app.command(
    help="DeepChopper is All You Need: encode the given fastq",
)
def encode(data_folder: Path, *, chunk: bool = False, chunk_size: int = 1000000, parallel: bool = False):
    """Encode the given fastq files to parquet format."""
    if not data_folder.exists():
        msg = f"Folder {data_folder} does not exist."
        logging.error(msg)

    fq_files = (
        [data_folder] if data_folder.is_file() else list(data_folder.glob("*.fq")) + list(data_folder.glob("*.fastq"))
    )

    for fq_file in fq_files:
        logging.info(f"Processing {fq_file}")
        if not chunk:
            encode_fq_path_to_parquet(
                fq_file,
                default.KMER_SIZE,
                bases=default.BASES,
                qual_offset=default.QUAL_OFFSET,
                vectorized_target=default.VECTORIZED_TARGET,
            )
        else:
            encode_fq_path_to_parquet_chunk(
                fq_file,
                chunk_size=chunk_size,
                parallel=parallel,
                bases=default.BASES,
                qual_offset=default.QUAL_OFFSET,
                vectorized_target=default.VECTORIZED_TARGET,
            )


@app.command(
    help="DeepChopper is All You Need",
)
def predict(
    data_path: Path,
    gpus: int = 0,
    output_path: Path | None = None,
    batch_size: int = 12,
    num_workers: int = 0,
    max_sample: int | None = None,
    *,
    limit_predict_batches: int | None = None,
):
    """Predict the given dataset using the given model and tokenizer."""
    tokenizer = deepchopper.models.llm.load_tokenizer_from_hyena_model(model_name="hyenadna-small-32k-seqlen")

    datamodule: LightningDataModule = deepchopper.data.fq_datamodule.FqDataModule(
        train_data_path="dummy.parquet",
        tokenizer=tokenizer,
        predict_data_path=data_path.as_posix(),
        batch_size=batch_size,
        num_workers=num_workers,
        max_predict_samples=max_sample,
    )

    model = deepchopper.DeepChopper.from_pretrained("yangliz5/deepchopper")

    output_path = Path(output_path or "predictions")

    callbacks = [deepchopper.models.callbacks.CustomWriter(output_dir=output_path, write_interval="batch")]

    available_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0

    if gpus > 0 and available_gpus > 0:
        accelerator = "gpu"
        if gpus > available_gpus:
            logging.warning(f"Number of GPUs requested: {gpus} is greater than available GPUs: {available_gpus}")
            gpus = available_gpus
        elif gpus < available_gpus:
            logging.info(f"Using {gpus} out of {available_gpus} GPUs for prediction.")
    else:
        accelerator = "cpu"
        gpus = "auto"

    trainer = lightning.pytorch.trainer.Trainer(
        accelerator=accelerator,
        devices=gpus,
        callbacks=callbacks,
        deterministic=False,
        logger=False,
        limit_predict_batches=limit_predict_batches,
    )

    trainer.predict(model=model, dataloaders=datamodule, return_predictions=False)


@app.command(
    help="DeepChopper is All You Need: chop your reads!",
)
def chop(
    predicts: list[Path],
    fq: Path,
    smooth_window_size: int = 21,
    min_interval_size: int = 13,
    approved_interval_number: int = 20,
    max_process_intervals: int = 4,
    min_read_length_after_chop: int = 20,
    output_chopped_seqs: Annotated[bool, typer.Option(help="if outupt chopped seqs")] = False,
    chop_type: str = "all",
    threads: int = 2,
    output_prefix: str | None = None,
    max_batch_size: int | None = None,
):
    from shutil import which

    if which("deepchopper-chop") is None:
        print("deepchopper-chop is not installed. Please use `cargo install deepchopper-chop` to install it.")
        raise SystemExit

    import subprocess

    predict_files = " ".join([f"--pdt {predict}" for predict in predicts])

    commands = [
        [
            "deepchopper-chop",
            predict_files,
            "--fq",
            fq,
            "-t",
            threads,
            "-s",
            smooth_window_size,
            "--mis",
            min_interval_size,
            "-a",
            approved_interval_number,
            "--mpi",
            max_process_intervals,
            "--mcr",
            min_read_length_after_chop,
            "--ocq",
            output_chopped_seqs,
            "--ct",
            chop_type,
            "-o",
            output_prefix,
            "-m",
            max_batch_size,
        ],
    ]
    subprocess.run(commands, check=True)


@app.command(
    help="DeepChopper is All You Need: ui!",
)
def web():
    """Run the web interface."""
    deepchopper.ui.main()


if __name__ == "__main__":
    app()
