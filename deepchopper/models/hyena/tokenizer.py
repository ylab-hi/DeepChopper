from functools import partial

import torch
from transformers import (
    AutoTokenizer,
    DataCollatorForTokenClassification,
)

import deepchopper


def pad_without_fast_tokenizer_warning(tokenizer, *pad_args, **pad_kwargs):
    """Pads without triggering the warning about how using the pad function is sub-optimal when using a fast tokenizer."""
    # To avoid errors when using Feature extractors
    if not hasattr(tokenizer, "deprecation_warnings"):
        return tokenizer.pad(*pad_args, **pad_kwargs)

    # Save the state of the warning, then disable it
    warning_state = tokenizer.deprecation_warnings.get("Asking-to-pad-a-fast-tokenizer", False)
    tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True

    try:
        padded = tokenizer.pad(*pad_args, **pad_kwargs)
    finally:
        # Restore the state of the warning.
        tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = warning_state

    return padded


class DataCollatorForTokenClassificationWithQual(DataCollatorForTokenClassification):
    def torch_call(self, features):
        import torch

        label_name = "label" if "label" in features[0] else "labels"
        labels = (
            [feature[label_name] for feature in features] if label_name in features[0] else None
        )

        qual_name = "input_quals"
        qual_pad_token_id = 0
        input_quals = [feature[qual_name] for feature in features]

        no_labels_features = [
            {k: v for k, v in feature.items() if k not in [qual_name, label_name]}
            for feature in features
        ]

        batch = pad_without_fast_tokenizer_warning(
            self.tokenizer,
            no_labels_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        if labels is None:
            return batch

        sequence_length = batch["input_ids"].shape[1]
        padding_side = self.tokenizer.padding_side

        def to_list(tensor_or_iterable):
            if isinstance(tensor_or_iterable, torch.Tensor):
                return tensor_or_iterable.tolist()
            return list(tensor_or_iterable)

        if padding_side == "right":
            batch[label_name] = [
                to_list(label) + [self.label_pad_token_id] * (sequence_length - len(label))
                for label in labels
            ]
            batch[qual_name] = [
                to_list(qual) + [qual_pad_token_id] * (sequence_length - len(qual))
                for qual in input_quals
            ]
        else:
            batch[label_name] = [
                [self.label_pad_token_id] * (sequence_length - len(label)) + to_list(label)
                for label in labels
            ]
            batch[qual_name] = [
                [qual_pad_token_id] * (sequence_length - len(qual)) + to_list(qual)
                for qual in input_quals
            ]

        batch[label_name] = torch.tensor(batch[label_name], dtype=torch.int64)
        batch[qual_name] = torch.tensor(batch[qual_name], dtype=torch.int64)
        return batch


def load_tokenizer_from_hyena_model(model_name):
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
    return AutoTokenizer.from_pretrained(
        model_name, max_length=max_length, truncation=True, padding=True, trust_remote_code=True
    )


def tokenize_and_align_labels_and_quals(data, tokenizer, max_length, pad_qual=0, pad_label=-100):
    tokenized_inputs = tokenizer(data["seq"], max_length=max_length, truncation=True, padding=True)
    labels = torch.tensor(
        [*deepchopper.vertorize_target(*data["target"], len(data["seq"])), pad_label]
    )
    quals = torch.cat((data["qual"], torch.tensor([pad_qual]))).float()
    torch.nn.functional.normalize(quals, dim=0)
    tokenized_inputs.update({"labels": labels, "input_quals": quals})
    return tokenized_inputs


def tokenize_dataset(dataset, tokenizer, max_length):
    return dataset.map(
        partial(tokenize_and_align_labels_and_quals, tokenizer=tokenizer, max_length=max_length)
    ).remove_columns(["id", "seq", "qual", "target"])