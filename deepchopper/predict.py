import multiprocessing
from functools import partial
from pathlib import Path

import typer
from datasets import load_dataset
from rich import print
from torch import Tensor
from transformers import AutoTokenizer, Trainer, TrainingArguments

from deepchopper.data import encode_one_fq_file
from deepchopper.models.hyena import (
    DataCollatorForTokenClassificationWithQual,
    TokenClassification,
    compute_metrics,
    tokenize_and_align_labels_and_quals,
)
from deepchopper.utils import (
    alignment_predict,
    highlight_target,
    summary_predict,
)

from .deepchopper import (
    write_predicts,
    remove_intervals_and_keep_left,
    smooth_label_region,
)
from .utils import hightlight_predicts


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


def load_dataset_and_model(check_point: Path, data_path: Path, max_sample: int = 500):
    """Load the dataset and model from the given paths."""
    if isinstance(check_point, str):
        check_point = Path(check_point)

    if isinstance(data_path, str):
        data_path = Path(data_path)

    if data_path.suffix in (".fq", ".fastq"):
        encode_one_fq_file(data_path)
        data_path = data_path.with_suffix(".parquet")

    if data_path.suffix != ".parquet":
        msg = f"Unsupported file format: {data_path.suffix}"
        raise ValueError(msg)

    resume_tokenizer = AutoTokenizer.from_pretrained(check_point, trust_remote_code=True)
    resume_model = TokenClassification.from_pretrained(check_point)

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


def load_trainer(resume_tokenizer, resume_model, batch_size: int = 24):
    """Load the trainer with the given model and tokenizer."""
    data_collator = DataCollatorForTokenClassificationWithQual(resume_tokenizer)
    training_args = TrainingArguments(
        output_dir="hyena_model_use_qual_testt",
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

    # Initialize our Trainer
    return Trainer(
        model=resume_model,
        args=training_args,
        tokenizer=resume_tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )


app = typer.Typer()


@app.command()
def main(
    check_point: Path,
    data_path: Path,
    max_sample: int = 1000,
    min_region_length_for_smooth: int = 1,
    max_distance_for_smooth: int = 1,
    *,
    show_sample: bool = False,
    save_predict: bool = False,
):
    """Predict the given dataset using the given model and tokenizer."""
    eval_dataset, tokenized_eval_dataset, resume_tokenizer, resume_model = load_dataset_and_model(
        check_point, data_path, max_sample
    )

    if show_sample:
        random_show_seq(eval_dataset, sample=3)

    trainer = load_trainer(resume_tokenizer, resume_model)
    predicts = trainer.predict(tokenized_eval_dataset)  # type: ignore
    print(predicts[2])

    true_prediction, true_label = summary_predict(predictions=predicts[0], labels=predicts[1])
    if show_sample:
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
        outout_path = data_path.with_suffix(".chopped.fq.gz")
        write_predicts(
            data_path,
            outout_path,
            true_prediction,
            min_region_length_for_smooth,
            max_distance_for_smooth,
        )


if __name__ == "__main__":
    app()
