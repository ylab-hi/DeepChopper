import multiprocessing
from functools import partial

import gradio as gr
import lightning
import torch
from datasets import Dataset

import deepchopper
from deepchopper.deepchopper import default, encode_qual, remove_intervals_and_keep_left, smooth_label_region
from deepchopper.models.llm import (
    tokenize_and_align_labels_and_quals,
)
from deepchopper.utils import (
    summary_predict,
)


def parse_fq_record(text: str):
    lines = text.strip().split("\n")
    for i in range(0, len(lines), 4):
        content = lines[i : i + 4]
        record_id, seq, _, qual = content
        yield {
            "id": record_id,
            "seq": seq,
            "qual": encode_qual(qual, default.KMER_SIZE),
            "target": [0, 0],
        }


def load_dataset(text: str, tokenizer):
    dataset = Dataset.from_generator(parse_fq_record, gen_kwargs={"text": text}).with_format("torch")
    tokenized_dataset = dataset.map(
        partial(
            tokenize_and_align_labels_and_quals,
            tokenizer=tokenizer,
            max_length=tokenizer.max_len_single_sentence,
        ),
        num_proc=multiprocessing.cpu_count(),  # type: ignore
    ).remove_columns(["id", "seq", "qual", "target"])

    return dataset, tokenized_dataset


def predict(
    text: str,
    smooth_window_size: int = 21,
    min_interval_size: int = 13,
    approved_interval_number: int = 20,
    max_process_intervals: int = 4,
    batch_size: int = 12,
    num_workers: int = 1,
):
    print(text)
    tokenizer = deepchopper.models.llm.load_tokenizer_from_hyena_model(model_name="hyenadna-small-32k-seqlen")
    dataset, tokenized_dataset = load_dataset(text, tokenizer)

    from torch.utils.data import DataLoader

    dataloader = DataLoader(tokenized_dataset, batch_size=batch_size, num_workers=num_workers)

    model = deepchopper.DeepChopper.from_pretrained("yangliz5/deepchopper")

    accelerator = "cpu" if torch.cuda.is_available() else "gpu"
    trainer = lightning.pytorch.trainer.Trainer(
        accelerator=accelerator,
        devices=-1,
        deterministic=False,
        logger=False,
    )

    predicts = trainer.predict(model=model, dataloaders=dataloader, return_predictions=True)

    assert len(predicts) == 1

    for idx, preds in enumerate(predicts):
        highted_text = []
        total_intervals = []

        true_prediction, _true_label = summary_predict(predictions=preds[0], labels=preds[1])

        _id = dataset[idx]["id"]
        seq = dataset[idx]["seq"]

        smooth_predict_targets = smooth_label_region(
            true_prediction[0], smooth_window_size, min_interval_size, approved_interval_number
        )

        if not smooth_predict_targets or len(smooth_predict_targets) > max_process_intervals:
            continue

        # zip two consecutive elements
        _selected_seqs, _selected_intervals = remove_intervals_and_keep_left(seq, smooth_predict_targets)

        total_intervals.extend(_selected_intervals)
        total_intervals.extend(smooth_predict_targets)

        if total_intervals:
            total_intervals.sort()
            for interval in total_intervals:
                if interval in smooth_predict_targets:
                    highted_text.append((seq[interval[0] : interval[1]], "ada"))
                else:
                    highted_text.append((seq[interval[0] : interval[1]], None))

    return highted_text


def main():
    demo = gr.Interface(
        fn=predict,
        inputs=[
            gr.Textbox(label="Input Text"),
        ],
        outputs=[gr.JSON(label="JSON Output"), gr.HighlightedText(label="Highlighted Text")],
        examples=[["example text"]],  # Add an example if you have one
    )
    demo.launch(debug=True)
