import logging
from pathlib import Path
from typing import Annotated

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
    predict_cli,
)
from .utils import (
    highlight_target,
)

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
    # check_point: Path,
    data_path: Path,
    output_path: Path | None = None,
    batch_size: int = 12,
    num_workers: int = 0,
    max_sample: int | None = None,
    *,
    show_sample: Annotated[bool, typer.Option(help="if show sample")] = False,
    limit_predict_batches: int | None = None,
):
    """Predict the given dataset using the given model and tokenizer."""
    tokenizer = deepchopper.models.llm.load_tokenizer_from_hyena_model(model_name="hyenadna-small-32k-seqlen")

    datamodule: LightningDataModule = deepchopper.data.fq_datamodule.FqDataModule(
        train_data_path="dummy.parquet",
        tokenizer=tokenizer,
        predict_data_path=data_path,
        batch_size=batch_size,
        max_predict_samples=max_sample,
    )

    model = deepchopper.DeepChopper.from_pretrained("yangliz5/deepchopper")

    callbacks = [deepchopper.models.callbacks.CustomWriter(output_dir="predictions", write_interval="batch")]

    accelerator = "cpu" if torch.cuda.is_available() else "gpu"
    trainer = lightning.pytorch.trainer.Trainer(
        default_root_dir=".",
        accelerator=accelerator,
        devices=-1,
        callbacks=callbacks,
        deterministic=False,
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
    print(predicts)
    print(fq)
    predict_cli(
        predicts,
        fq,
        smooth_window_size,
        min_interval_size,
        approved_interval_number,
        max_process_intervals,
        min_read_length_after_chop,
        output_chopped_seqs,
        chop_type,
        threads,
        output_prefix,
        max_batch_size,
    )


@app.command(
    help="DeepChopper is All You Need: ui!",
)
def ui():
    deepchopper.ui.main()


if __name__ == "__main__":
    app()
