import logging
import multiprocessing
from functools import partial
from pathlib import Path
from typing import Annotated

import typer
from datasets import load_dataset
from rich import print
from rich.logging import RichHandler
from torch import Tensor
from transformers import AutoTokenizer, Trainer, TrainingArguments

from .deepchopper import (
    convert_multiple_fqs_to_one_fq,
    default,
    encode_fq_path_to_parquet,
    encode_fq_path_to_parquet_chunk,
    remove_intervals_and_keep_left,
    smooth_label_region,
    write_predicts,
)
from .models.hyena import (
    DataCollatorForTokenClassificationWithQual,
    TokenClassification,
    compute_metrics,
    tokenize_and_align_labels_and_quals,
)
from .utils import (
    alignment_predict,
    highlight_target,
    hightlight_predicts,
    summary_predict,
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


def load_dataset_and_model(check_point: Path, data_path: Path, max_sample: int | None = None):
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

    resume_tokenizer = AutoTokenizer.from_pretrained(check_point, trust_remote_code=True)
    resume_model = TokenClassification.from_pretrained(check_point)

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

    return eval_dataset, tokenized_eval_dataset, resume_tokenizer, resume_model


def load_trainer(
    resume_tokenizer, resume_model, batch_size: int = 24, *, show_metrics: bool = False
):
    """Load the trainer with the given model and tokenizer."""
    data_collator = DataCollatorForTokenClassificationWithQual(resume_tokenizer)
    training_args = TrainingArguments(
        output_dir="deepchopper",
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
    help="DeepChopper is All You Need",
)
def predict(
    check_point: Path,
    data_path: Path,
    output_path: Path | None = None,
    batch_size: int = 12,
    max_sample: int | None = None,
    min_region_length_for_smooth: int = 1,
    max_distance_for_smooth: Annotated[int, typer.Option()] = 1,
    *,
    show_metrics: Annotated[bool, typer.Option(help="if show metrics")] = False,
    show_sample: Annotated[bool, typer.Option(help="if show sample")] = False,
    save_predict: Annotated[bool, typer.Option(help="if save predict")] = False,
):
    """Predict the given dataset using the given model and tokenizer."""
    eval_dataset, tokenized_eval_dataset, resume_tokenizer, resume_model = load_dataset_and_model(
        check_point, data_path, max_sample
    )

    if show_sample:
        random_show_seq(eval_dataset, sample=3)

    trainer = load_trainer(
        resume_tokenizer, resume_model, batch_size=batch_size, show_metrics=show_metrics
    )
    predicts = trainer.predict(tokenized_eval_dataset)  # type: ignore

    true_prediction, true_label = summary_predict(predictions=predicts[0], labels=predicts[1])
    if show_sample:
        print(predicts[2])
        alignment_predict(true_prediction[0], true_label[0])
        for idx, preds in enumerate(true_prediction):
            record_id = eval_dataset[idx]["id"]
            seq = eval_dataset[idx]["seq"]
            smooth_predict_targets = smooth_label_region(
                preds, min_region_length_for_smooth, max_distance_for_smooth
            )

            targets = eval_dataset[idx]["target"]

            if isinstance(targets, Tensor):
                targets = targets.tolist()
            # zip two consecutive elements
            targets = [(targets[i], targets[i + 1]) for i in range(0, len(targets), 2)]
            print(f"{record_id=}")
            hightlight_predicts(seq, targets, smooth_predict_targets)

            _selected_seqs, _selected_intervals = remove_intervals_and_keep_left(
                seq, smooth_predict_targets
            )
    elif save_predict:
        if output_path is None:
            outout_path = data_path.with_suffix(".chopped.fq.gz")
        else:
            if not output_path.exists():
                output_path.mkdir(parents=True, exist_ok=True)
            outout_path = output_path / data_path.with_suffix(".chopped.fq.gz").name

        write_predicts(
            data_path,
            outout_path,
            true_prediction,
            min_region_length_for_smooth,
            max_distance_for_smooth,
        )

    if (Path.cwd() / "deepchopper").exists():
        (Path.cwd() / "deepchopper").rmdir()


@app.command(
    help="DeepChopper is All You Need: encode the given fastq",
)
def encode(
    data_folder: Path, *, chunk: bool = False, chunk_size: int = 1000000, parallel: bool = False
):
    """Encode the given fastq files to parquet format."""
    if not data_folder.exists():
        msg = f"Folder {data_folder} does not exist."
        logging.error(msg)

    fq_files = (
        [data_folder]
        if data_folder.is_file()
        else list(data_folder.glob("*.fq")) + list(data_folder.glob("*.fastq"))
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


if __name__ == "__main__":
    app()
