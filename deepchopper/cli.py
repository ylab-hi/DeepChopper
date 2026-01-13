import logging
from pathlib import Path
from typing import TYPE_CHECKING

import lightning
import torch
import typer
from click import Context
from rich import print
from rich.logging import RichHandler
from typer.core import TyperGroup

import deepchopper

from .utils import (
    highlight_target,
)

if TYPE_CHECKING:
    from lightning.pytorch import LightningDataModule


def set_logging_level(level: int = logging.INFO):
    """Set the logging level.

    Parameters:
        level (int): The logging level to set.
    """
    FORMAT = "%(message)s"
    logging.basicConfig(
        level=level,
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


def encode(
    fastq_path: Path = typer.Argument(None, help="DEPRECATED: Use 'deepchopper predict' instead"),
):
    """DEPRECATED: Please use `deepchopper predict fastq_path` directly."""
    typer.secho(
        "❌ Error: The 'encode' command is deprecated.\n   Please use 'deepchopper predict <fastq_path>' instead.",
        fg=typer.colors.RED,
        err=True,
    )
    raise typer.Exit(1)


def predict(
    data_path: Path = typer.Argument(..., help="Path to the dataset"),
    gpus: int = typer.Option(0, "--gpus", "-g", help="Number of GPUs to use"),
    output_path: Path | None = typer.Option(None, "--output", "-o", help="Output path for predictions"),
    batch_size: int = typer.Option(12, "--batch-size", "-b", help="Batch size"),
    num_workers: int = typer.Option(0, "--workers", "-w", help="Number of workers"),
    model: str = typer.Option(
        "rna002",
        "--model",
        "-m",
        help="Model name (choices: rna002, rna004)",
        show_choices=True,
        case_sensitive=False,
        metavar="MODEL",
        rich_help_panel="Model",
        callback=lambda v: v.lower()
        if v.lower() in {"rna002", "rna004"}
        else typer.BadParameter("Model must be one of: rna002, rna004"),
    ),
    limit_predict_batches: int | None = typer.Option(None, "--limit-batches", help="Limit prediction batches"),
    max_sample: int | None = typer.Option(None, "--max-sample", help="Maximum number of samples to process"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output"),
):
    """Predict the given dataset using DeepChopper."""
    if verbose:
        set_logging_level(logging.INFO)

    # Path validation
    if isinstance(data_path, str):
        data_path = Path(data_path)

    if not data_path.exists():
        typer.secho(f"❌ Error: Data path '{data_path}' does not exist.", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)

    # Set seed only after validation passes
    lightning.seed_everything(42, workers=True)

    tokenizer = deepchopper.models.llm.load_tokenizer_from_hyena_model(model_name="hyenadna-small-32k-seqlen")
    datamodule: LightningDataModule = deepchopper.data.OnlyFqDataModule(
        train_data_path="dummy.parquet",
        tokenizer=tokenizer,
        predict_data_path=data_path.as_posix(),
        batch_size=batch_size,
        num_workers=num_workers,
        max_predict_samples=max_sample,
    )

    model = (
        deepchopper.DeepChopper.from_pretrained("yangliz5/deepchopper")
        if model == "rna002"
        else deepchopper.DeepChopper.from_pretrained("yangliz5/deepchopper-rna004")
    )
    output_path = Path(output_path or "predictions")
    callbacks = [deepchopper.models.callbacks.CustomWriter(output_dir=output_path, write_interval="batch")]

    if gpus > 0:
        if torch.cuda.is_available():
            accelerator = "gpu"
            devices = min(gpus, torch.cuda.device_count())
        elif torch.backends.mps.is_available():
            accelerator = "mps"
            devices = "auto"  # MPS currently supports only one device
        else:
            accelerator = "cpu"
            devices = "auto"
    else:
        accelerator = "cpu"
        devices = "auto"

    trainer = lightning.pytorch.trainer.Trainer(
        accelerator=accelerator,
        devices=devices,
        callbacks=callbacks,
        deterministic=True,
        logger=False,
        limit_predict_batches=limit_predict_batches,
    )

    import multiprocess.context as ctx

    ctx._force_start_method("spawn")
    trainer.predict(model=model, dataloaders=datamodule, return_predictions=False)


def chop(
    predicts: list[Path] = typer.Argument(..., help="Paths to prediction files"),
    fq: Path = typer.Argument(..., help="Path to FASTQ file"),
    smooth_window_size: int = typer.Option(21, "--smooth-window", help="Smooth window size"),
    min_interval_size: int = typer.Option(13, "--min-interval-size", help="Minimum interval size"),
    approved_interval_number: int = typer.Option(20, "--approved-intervals", help="Number of approved intervals"),
    max_process_intervals: int = typer.Option(4, "--max-process-intervals", help="Maximum process intervals"),
    min_read_length_after_chop: int = typer.Option(20, "--min-read-length", help="Minimum read length after chop"),
    output_chopped_seqs: bool = typer.Option(False, "--output-chopped", help="Output chopped sequences"),
    chop_type: str = typer.Option("all", "--chop-type", help="Chop type"),
    threads: int = typer.Option(2, "--threads", help="Number of threads"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output"),
    output_prefix: str | None = typer.Option(None, "--prefix", "-o", help="Output prefix"),
    max_batch_size: int | None = typer.Option(None, "--max-batch", help="Maximum batch size"),
):
    """Chop sequences based on predictions."""
    if verbose:
        set_logging_level(logging.INFO)

    import subprocess
    from shutil import which

    if which("deepchopper-chop") is None:
        print("deepchopper-chop is not installed. Please use `cargo install deepchopper-chop` to install it.")
        raise SystemExit

    predict_files = " ".join([f"--pdt {predict}" for predict in predicts])

    command = f"deepchopper-chop {predict_files} --fq {fq} -t {threads} -s {smooth_window_size} --mis {min_interval_size} -a {approved_interval_number} --mpi {max_process_intervals} --mcr {min_read_length_after_chop} --ct {chop_type} "

    if output_chopped_seqs:
        command += "--ocq "

    if output_prefix is not None:
        command += f"-o {output_prefix} "

    if max_batch_size is not None:
        command += f"-m {max_batch_size} "

    try:
        subprocess.run(command.split(), check=True)
    except subprocess.CalledProcessError as e:
        logging.error(f"Error: Chopping failed with exit code {e.returncode}")
        raise e


def web():
    """Run the web interface."""
    deepchopper.ui.main()


class OrderCommands(TyperGroup):
    """Order commands in the order appear."""

    def list_commands(self, ctx: Context):
        """Return list of commands in the order appear."""
        return list(self.commands)  # get commands using self.commands


def version_callback(value: bool):
    """Print the version and exit."""
    if value:
        print(f"DeepChopper Version: {deepchopper.__version__}")
        raise typer.Exit()


app = typer.Typer(
    cls=OrderCommands,
    context_settings={"help_option_names": ["-h", "--help"]},
    help="DeepChopper: A genomic lanuage model to identify artificial sequences.",
)


# Add the version option to the main app
@app.callback()
def main(
    version: bool | None = typer.Option(
        None,
        "--version",
        "-V",
        help="Show the application's version and exit.",
        callback=version_callback,
        is_eager=True,
    ),
):
    """DeepChopper CLI."""


app.command(
    help="DeepChopper: encode the given fastq (DEPRECATED)",
    epilog="DEPRECATED: Please use `deepchopper predict fastq_path` directly.",
)(encode)

app.command(
    help="DeepChopper: predict the given dataset",
    epilog="Example: deepchopper predict fastq_path --gpus 1 --output predictions",
)(predict)

app.command(
    help="DeepChopper: chop the given predictions!",
    epilog="Example: deepchopper chop predictions/0 fastq_path",
)(chop)

app.command(
    help="DeepChopper: a web ui!",
    epilog="Example: deepchopper web",
)(web)


if __name__ == "__main__":
    app()
