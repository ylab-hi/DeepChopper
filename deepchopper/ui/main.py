import multiprocessing
from functools import partial
from pathlib import Path

import gradio as gr
from datasets import Dataset

from deepchopper.deepchopper import encode_qual, remove_intervals_and_keep_left, smooth_label_region
from deepchopper.models.hyena import (
    tokenize_and_align_labels_and_quals,
)
from deepchopper.predict import (
    load_model_from_checkpoint,
    load_trainer,
)
from deepchopper.utils import (
    highlight_targets,
    summary_predict,
)

check_point = Path.cwd() / "deepchopper_train"
tokenizer, model = load_model_from_checkpoint(check_point)
trainer = load_trainer(tokenizer, model, show_metrics=False)


def parse_fq_record(text: str):
    lines = text.strip().split("\n")
    for i in range(0, len(lines), 4):
        content = lines[i : i + 4]
        record_id, seq, _, qual = content
        yield {"id": record_id, "seq": seq, "qual": encode_qual(qual), "target": [0, 0]}


def load_dataset(text: str, tokenizer):
    dataset = Dataset.from_generator(parse_fq_record(text))
    tokenized_dataset = dataset.map(
        partial(
            tokenize_and_align_labels_and_quals,
            tokenizer=tokenizer,
            max_length=tokenizer.max_len_single_sentence,
        ),
        num_proc=multiprocessing.cpu_count(),  # type: ignore
    ).remove_columns(["id", "seq", "qual", "target"])
    return dataset, tokenized_dataset


def predict(text: str):
    min_region_length_for_smooth = 1
    max_distance_for_smooth = 1
    dataset, tokenized_dataset = load_dataset(text, tokenizer)
    predicts = trainer.predict(tokenized_dataset)  # type: ignore
    true_prediction, _true_label = summary_predict(predictions=predicts[0], labels=predicts[1])

    for idx, preds in enumerate(true_prediction):
        dataset[idx]["id"]
        seq = dataset[idx]["seq"]
        smooth_predict_targets = smooth_label_region(
            preds, min_region_length_for_smooth, max_distance_for_smooth
        )
        # zip two consecutive elements
        highlight_targets(seq, smooth_predict_targets)
        _selected_seqs, _selected_intervals = remove_intervals_and_keep_left(
            seq, smooth_predict_targets
        )

    return predicts[2]


def main():
    demo = gr.Interface(
        fn=predict,
        inputs=["text"],
        outputs=["text"],
    )

    demo.launch()
