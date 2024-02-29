from functools import partial

import torch
from transformers import (
    AutoConfig,
    AutoTokenizer,
)

import deepchopper


def load_config_and_tokenizer_from_hyena_model(model_name):
    max_lengths = {
        "hyenadna-tiny-1k-seqlen": 1024,
        "hyenadna-small-32k-seqlen": 32768,
        "hyenadna-medium-160k-seqlen": 160000,
        "hyenadna-medium-450k-seqlen": 450000,  # T4 up to here
        "hyenadna-large-1m-seqlen": 1_000_000,  # only A100 (paid tier)
    }

    if model_name not in max_lengths:
        msg = f"Model name {model_name} not found in available models."
        raise ValueError(msg)

    max_length = max_lengths[model_name]
    # bfloat16 for better speed and reduced memory usage
    model_name = f"LongSafari/{model_name}-hf"
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, max_length=max_length, truncation=True, padding=True, trust_remote_code=True
    )
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    return tokenizer, config


def tokenize_and_align_labels_and_quals(data, tokenizer, max_length, pad_qual=0, pad_label=-100):
    tokenized_inputs = tokenizer(data["seq"], max_length=max_length, truncation=True, padding=True)
    labels = torch.tensor(
        [*deepchopper.vertorize_target(*data["target"], len(data["seq"])), pad_label]
    )
    quals = torch.cat((data["qual"], torch.tensor([pad_qual])))
    tokenized_inputs.update({"labels": labels, "input_quals": quals})
    return tokenized_inputs


def tokenize_dataset(dataset, tokenizer, max_length):
    return dataset.map(
        partial(tokenize_and_align_labels_and_quals, tokenizer=tokenizer, max_length=max_length)
    ).remove_columns(["id", "seq", "qual", "target"])
