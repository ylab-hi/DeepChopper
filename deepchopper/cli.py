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
    check_point: Path,
    data_path: Path,
    output_path: Path | None = None,
    batch_size: int = 12,
    max_sample: int | None = None,
    *,
    show_metrics: Annotated[bool, typer.Option(help="if show metrics")] = False,
    show_sample: Annotated[bool, typer.Option(help="if show sample")] = False,
    save_predict: Annotated[bool, typer.Option(help="if save predict")] = False,
):
    """Predict the given dataset using the given model and tokenizer."""
    tokenizer = deepchopper.models.llm.load_tokenizer_from_hyena_model(model_name="hyenadna-small-32k-seqlen")

    datamodule: LightningDataModule = deepchopper.data.fq_datamodule.FqDataModule(
        train_data_path="test.parquet",
        tokenizer=tokenizer,
        predict_data_path=data_path,
        batch_size=batch_size,
        max_predict_samples=max_sample,
    )

    model = deepchopper.models.basic_module.TokenClassificationLit.load_from_checkpoint(
        checkpoint_path=check_point,
        net=deepchopper.models.llm.hyena.TokenClassificationModule(
            number_of_classes=2,
            backbone_name="hyenadna-small-32k-seqlen",
            freeze_backbone=False,
            head=deepchopper.models.llm.TokenClassificationHead(
                input_size=256,
                lin1_size=1024,
                lin2_size=1024,
                num_class=2,
                use_identity_layer_for_qual=True,
                use_qual=True,
            ),
        ),
        criterion=deepchopper.models.basic_module.ContinuousIntervalLoss(lambda_penalty=0),
    )

    accelerator = "cpu" if torch.cuda.is_available() else "gpu"

    trainer = lightning.pytorch.trainer.Trainer(
        default_root_dir=".",
        accelerator=accelerator,
        devices=-1,
        deterministic=False,
    )
    print(model)

    # import multiprocess.context as ctx
    # ctx._force_start_method("spawn")
    # trainer.predict(model=model, dataloaders=datamodule, return_predictions=False)


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


if __name__ == "__main__":
    app()
