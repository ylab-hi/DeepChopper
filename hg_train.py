import logging
import multiprocessing
import os
import sys
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path

import datasets
import numpy as np
import torch
import transformers
from datasets import Dataset, load_dataset
from transformers import (
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import send_example_telemetry

from deepchopper.data import load_and_split_dataset
from deepchopper.models.llm import (
    IGNORE_INDEX,
    DataCollatorForTokenClassificationWithQual,
    HyenadnaMaxLengths,
    TokenClassification,
    TokenClassificationConfig,
    compute_metrics,
    tokenize_and_align_labels_and_quals,
)

logger = logging.getLogger(__name__)

torch.cuda.empty_cache()


@dataclass
class ModelArguments:
    """Arguments pertaining to which model/config/tokenizer we are going to fine-tune from."""

    model_name_or_path: str | None = field(
        default=None,
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        },
    )

    hyenadna_model: str = field(
        default="hyenadna-small-32k-seqlen",
        metadata={"help": "The name of the hyenadna model to use."},
    )

    config_name: str | None = field(
        default=None,
        metadata={"help": "Pretrained config name or path if not the same as model_name"},
    )
    tokenizer_name: str | None = field(
        default=None,
        metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"},
    )
    cache_dir: str | None = field(
        default=None,
        metadata={
            "help": "Where do you want to store the pretrained models downloaded from huggingface.co"
        },
    )
    model_revision: str = field(
        default="main",
        metadata={
            "help": "The specific model version to use (can be a branch name, tag name or commit id)."
        },
    )
    token: str | None = field(
        default=None,
        metadata={
            "help": (
                "The token to use as HTTP bearer authorization for remote files. If not specified, will use the token "
                "generated when running `huggingface-cli login` (stored in `~/.huggingface`)."
            )
        },
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether or not to allow for custom models defined on the Hub in their own modeling files. This option "
                "should only be set to `True` for repositories you trust and in which you have read the code, as it will "
                "execute code present on the Hub on your local machine."
            )
        },
    )
    ignore_mismatched_sizes: bool = field(
        default=False,
        metadata={
            "help": "Will enable to load a pretrained model whose head dimensions are different."
        },
    )

    def __post_init__(self):
        if self.hyenadna_model not in HyenadnaMaxLengths:
            msg = (
                f"Invalid hyenadna model name: {self.hyenadna_model}. "
                f"Should be one of {list(HyenadnaMaxLengths.keys())}"
            )
            raise ValueError(msg)


@dataclass
class DataTrainingArguments:
    """Arguments pertaining to what data we are going to input our model for training and eval."""

    task_name: str | None = field(
        default="deepchopper", metadata={"help": "The name of the task (ner, pos...)."}
    )
    dataset_name: str | None = field(
        default=None,
        metadata={"help": "The name of the dataset to use (via the datasets library)."},
    )
    dataset_config_name: str | None = field(
        default=None,
        metadata={
            "help": "The configuration name of the dataset to use (via the datasets library)."
        },
    )
    train_file: str | None = field(
        default=None, metadata={"help": "The input training data file (a parquet file)."}
    )
    validation_file: str | None = field(
        default=None,
        metadata={
            "help": "An optional input evaluation data file to evaluate on (a parquet file)."
        },
    )
    test_file: str | None = field(
        default=None,
        metadata={"help": "An optional input test data file to predict on (a parquet file)."},
    )
    text_column_name: str | None = field(
        default=None,
        metadata={"help": "The column name of text to input in the file (a parquet file)."},
    )
    label_column_name: str | None = field(
        default=None,
        metadata={"help": "The column name of label to input in the file (a parquet file)."},
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: int | None = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_seq_length: int | None = field(
        default=None,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. If set, sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to pad all samples to model maximum sentence length. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
                "efficient on GPU but very bad for TPU."
            )
        },
    )
    max_train_samples: int | None = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: int | None = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    max_predict_samples: int | None = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                "value if set."
            )
        },
    )
    label_all_tokens: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to put the label for one word on all tokens of generated by that word or just on the "
                "one (in which case the other tokens will have a padding index)."
            )
        },
    )
    return_entity_level_metrics: bool = field(
        default=False,
        metadata={
            "help": "Whether to return all the entity levels during evaluation or just the overall ones."
        },
    )

    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")

        if self.train_file is not None:
            extension = self.train_file.split(".")[-1]
            if extension not in ["parquet"]:
                raise ValueError("`train_file` should be a parquet file.")

        if self.validation_file is not None:
            extension = self.validation_file.split(".")[-1]
            if extension not in ["parquet"]:
                raise ValueError("`validation_file` should be a parquet file.")

        if self.task_name is not None:
            self.task_name = self.task_name.lower()


def train():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=Path(sys.argv[1]).resolve().as_posix()
        )
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("deepchopper", model_args, data_args)

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if (
        Path(training_args.output_dir).is_dir()
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            msg = (
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
            raise ValueError(msg)

        if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.

    extension = "parquet"
    if data_args.dataset_name is not None:
        single_dataset = True
        # Downloading and loading a dataset from the hub.
        # split "train", "validation", "test" from a dataset
        # mainly used for small datasets for testing
        train_dataset, eval_dataset, predict_dataset = load_and_split_dataset(
            data_args.dataset_name
        )
    else:
        num_proc = multiprocessing.cpu_count()
        single_dataset = False
        data_files = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file

        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file

        if data_args.test_file is not None:
            data_files["test"] = data_args.test_file

        raw_datasets = load_dataset(
            extension, data_files=data_files, num_proc=num_proc, cache_dir=model_args.cache_dir
        ).with_format("torch")

    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.

    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    if model_args.tokenizer_name is None:
        logger.info(f"config_name is None, loading {model_args.hyenadna_model} from hg")
        if model_args.hyenadna_model not in HyenadnaMaxLengths:
            msg = f"Model name {model_args.hyenadna_model } not found in available models."
            raise ValueError(msg)
        max_length = HyenadnaMaxLengths[model_args.hyenadna_model]
        hg_model = f"LongSafari/{model_args.hyenadna_model}-hf"
        tokenizer = AutoTokenizer.from_pretrained(
            hg_model,
            max_length=max_length,
            truncation=True,
            padding=True,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            token=model_args.token,
            trust_remote_code=model_args.trust_remote_code,
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.config_name,
            finetuning_task=data_args.task_name,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            token=model_args.token,
            trust_remote_code=model_args.trust_remote_code,
        )

    if model_args.config_name is not None:
        model = TokenClassification.from_pretrained(
            model_args.config_name,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            trust_remote_code=model_args.trust_remote_code,
            ignore_mismatched_sizes=model_args.ignore_mismatched_sizes,
        )
    else:
        model_config = TokenClassificationConfig()
        model = TokenClassification(model_config)

    # Preprocessing the dataset
    # Padding strategy
    if training_args.do_train:
        if not single_dataset and "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")

        if not single_dataset:
            train_dataset = raw_datasets["train"]

        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = Dataset.from_dict(train_dataset[:max_train_samples]).with_format(
                "torch"
            )

        with training_args.main_process_first(desc="train dataset map pre-processing"):
            train_dataset = (
                train_dataset.map(
                    partial(
                        tokenize_and_align_labels_and_quals,
                        tokenizer=tokenizer,
                        max_length=tokenizer.max_len_single_sentence,
                    ),
                    batched=False,
                    num_proc=data_args.preprocessing_num_workers,
                    load_from_cache_file=not data_args.overwrite_cache,
                    desc="Running tokenizer on train dataset",
                )
                .remove_columns(["id", "seq", "qual", "target"])
                .shuffle()
            )

    if training_args.do_eval:
        if not single_dataset and "validation" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")

        if not single_dataset:
            eval_dataset = raw_datasets["validation"]

        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = Dataset.from_dict(eval_dataset[:max_eval_samples]).with_format("torch")

        with training_args.main_process_first(desc="validation dataset map pre-processing"):
            eval_dataset = (
                eval_dataset.map(
                    partial(
                        tokenize_and_align_labels_and_quals,
                        tokenizer=tokenizer,
                        max_length=tokenizer.max_len_single_sentence,
                    ),
                    batched=False,
                    num_proc=data_args.preprocessing_num_workers,
                    load_from_cache_file=not data_args.overwrite_cache,
                    desc="Running tokenizer on validation dataset",
                )
                .remove_columns(["id", "seq", "qual", "target"])
                .shuffle()
            )

    if training_args.do_predict:
        if not single_dataset and "test" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")

        if not single_dataset:
            predict_dataset = raw_datasets["test"]

        if data_args.max_predict_samples is not None:
            max_predict_samples = min(len(predict_dataset), data_args.max_predict_samples)
            predict_dataset = Dataset.from_dict(predict_dataset[:max_predict_samples]).with_format(
                "torch"
            )

        with training_args.main_process_first(desc="prediction dataset map pre-processing"):
            predict_dataset = (
                predict_dataset.map(
                    partial(
                        tokenize_and_align_labels_and_quals,
                        tokenizer=tokenizer,
                        max_length=tokenizer.max_len_single_sentence,
                    ),
                    batched=False,
                    num_proc=data_args.preprocessing_num_workers,
                    load_from_cache_file=not data_args.overwrite_cache,
                    desc="Running tokenizer on prediction dataset",
                )
                .remove_columns(["id", "seq", "qual", "target"])
                .shuffle()
            )

    # Data collator
    data_collator = DataCollatorForTokenClassificationWithQual(
        tokenizer, pad_to_multiple_of=8 if training_args.fp16 else None
    )
    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # Training
    if training_args.do_train:
        checkpoint = None

        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint

        elif last_checkpoint is not None:
            checkpoint = last_checkpoint

        logger.info(f"{checkpoint=} {type(checkpoint)}")

        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        trainer.save_model()  # Saves the tokenizer too for easy upload

        max_train_samples = (
            data_args.max_train_samples
            if data_args.max_train_samples is not None
            else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate()

        max_eval_samples = (
            data_args.max_eval_samples
            if data_args.max_eval_samples is not None
            else len(eval_dataset)
        )
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Predict
    if training_args.do_predict:
        logger.info("*** Predict ***")

        predictions, labels, metrics = trainer.predict(predict_dataset, metric_key_prefix="predict")
        predictions = np.argmax(predictions, axis=2)

        # Remove ignored index (special tokens)
        true_predictions = [
            [p for (p, l) in zip(prediction, label, strict=True) if l != IGNORE_INDEX]
            for prediction, label in zip(predictions, labels, strict=True)
        ]

        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)

        # Save predictions
        output_predictions_file = Path(training_args.output_dir) / "predictions.txt"
        if trainer.is_world_process_zero():
            with Path(output_predictions_file).open("w") as writer:
                for prediction in true_predictions:
                    prediction_str = (str(x) for x in prediction)
                    writer.write(" ".join(prediction_str) + "\n")

    kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "token-classification"}
    if data_args.dataset_name is not None:
        kwargs["dataset_tags"] = data_args.dataset_name
        if data_args.dataset_config_name is not None:
            kwargs["dataset_args"] = data_args.dataset_config_name
            kwargs["dataset"] = f"{data_args.dataset_name} {data_args.dataset_config_name}"
        else:
            kwargs["dataset"] = data_args.dataset_name

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)


if __name__ == "__main__":
    train()
