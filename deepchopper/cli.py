import logging
import multiprocessing
from functools import partial
from pathlib import Path
from typing import Annotated

import typer
import torch
import lightning
from datasets import load_dataset
from rich import print
from rich.logging import RichHandler
from transformers import AutoTokenizer, Trainer, TrainingArguments

import deepchopper

from .deepchopper import (
    convert_multiple_fqs_to_one_fq,
    default,
    encode_fq_path_to_parquet,
    encode_fq_path_to_parquet_chunk,
    predict_cli,
)
from .models.llm import (
    DataCollatorForTokenClassificationWithQual,
    TokenClassification,
    compute_metrics,
    tokenize_and_align_labels_and_quals,
)
from .utils import (
    highlight_target,
    summary_predict,
)

FORMAT = "%(message)s"
logging.basicConfig(
    level=logging.INFO,
    format=FORMAT,
    handlers=[RichHandler()],
)

TMPOUTPUT = Path.cwd() / "deepchopper_predict"


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


def load_model_from_checkpoint(check_point: Path):
    """Load the model from the given path."""
    if isinstance(check_point, str):
        check_point = Path(check_point)
    resume_tokenizer = AutoTokenizer.from_pretrained(check_point, trust_remote_code=True)
    resume_model = TokenClassification.from_pretrained(check_point)
    return resume_tokenizer, resume_model


def load_dataset_from_checkpont(check_point: Path, data_path: Path, resume_tokenizer, max_sample: int | None = None):
    """Load the dataset and model from the given paths."""
    if isinstance(check_point, str):
        check_point = Path(check_point)

    if isinstance(data_path, str):
        data_path = Path(data_path)

    if data_path.suffix in (".fq", ".fastq"):
        msg = f"Please Encoded {data_path.with_suffix('.fq')} first using `deepchopper encode`"
        raise ValueError(msg)

    if data_path.suffix != ".parquet":
        msg = f"Unsupported file format: {data_path.suffix}"
        raise ValueError(msg)

    if max_sample is None:
        max_sample: str = "100%"

    eval_dataset = load_dataset(
        "parquet",
        data_files={"predict": data_path.as_posix()},
        num_proc=multiprocessing.cpu_count(),
        split=f"predict[:{max_sample}]",
    ).with_format("torch")

    tokenized_eval_dataset = eval_dataset.map(
        partial(
            tokenize_and_align_labels_and_quals,
            tokenizer=resume_tokenizer,
            max_length=resume_tokenizer.max_len_single_sentence,
        ),
        num_proc=multiprocessing.cpu_count(),  # type: ignore
    ).remove_columns(["id", "seq", "qual", "target"])

    return eval_dataset, tokenized_eval_dataset


def load_trainer(
    resume_tokenizer,
    resume_model,
    batch_size: int = 24,
    *,
    show_metrics: bool = False,
    use_cpu=False,
):
    """Load the trainer with the given model and tokenizer."""
    data_collator = DataCollatorForTokenClassificationWithQual(resume_tokenizer)
    training_args = TrainingArguments(
        output_dir=TMPOUTPUT.as_posix(),
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=1,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        push_to_hub=False,
        torch_compile=False,
        report_to="wandb",  # type: ignore
        run_name="eval",
        use_cpu=use_cpu,
    )

    compute_metrics_func = compute_metrics if show_metrics else None

    # Initialize our Trainer
    return Trainer(
        model=resume_model,
        args=training_args,
        tokenizer=resume_tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics_func,
    )


app = typer.Typer(
    context_settings={"help_option_names": ["-h", "--help"]},
)


@app.command(
    help="DeepChopper is All You Need: collect the given fastq",
)
def collect(data_folder: Path, output: Path):
    """Collect multiple fastq files to one fastq file."""
    if not data_folder.exists():
        msg = f"Folder {data_folder} does not exist."
        raise ValueError(msg)

    fq_files = (
        [data_folder]
        if data_folder.is_file()
        else list(data_folder.glob("*.fq"))
        + list(data_folder.glob("*.fastq"))
        + list(data_folder.glob("*.fq.gz"))
        + list(data_folder.glob("*.fastq.gz"))
    )
    convert_multiple_fqs_to_one_fq(fq_files, output)


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
    # if show_sample:
    #     random_show_seq(eval_dataset, sample=3)

    tokenizer = deepchopper.models.llm.load_tokenizer_from_hyena_model(model_name="hyenadna-small-32k-seqlen")

    datamodule: LightningDataModule = deepchopper.data.fq_data_module.FqDataModule(
        tokenizer=tokenizer,
        predict_data_path=data_path,
        batch_size=batch_size,
        max_predict_samples=max_sample,
    )
    model = deepchopper.models.basic_module.TokenClassificationLit(
        optimizer=torch.optim.Adam(lr=0.00002, weight_decay=0.0),
        scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(mode="min", factor=0.1, patience=10),
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
        accelerator=accelerator, w w
        devices=-1,
        deterministic=False,
    )

    import multiprocess.context as ctx

    ctx._force_start_method("spawn")
    trainer.predict(model=model, dataloaders=datamodule, ckpt_path=check_point, return_predictions=False)


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
