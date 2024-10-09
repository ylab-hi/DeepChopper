import typing

import numpy as np
import torch
from rich.console import Console
from rich.highlighter import RegexHighlighter
from rich.text import Text
from rich.theme import Theme

from deepchopper.deepchopper import summary_predict as rust_summary_predict
from deepchopper.models.llm import IGNORE_INDEX


def hightlight_predicts(
    seq: str,
    targets: list[tuple[int, int]],
    predicts: list[tuple[int, int]],
    style: str = "bold magenta",
    width: int = 80,
):
    """Highlight the predicted and labeled sequences."""
    target_seq = Text(seq)
    predict_seq = Text(seq)
    console = Console()

    for start, end in targets:
        target_seq.stylize(style, start, end)

    for start, end in predicts:
        predict_seq.stylize(style, start, end)

    front1 = "L:"
    front2 = "P:"
    for t1, t2 in zip(target_seq.wrap(console, width), predict_seq.wrap(console, width), strict=True):
        console.print(front1, t1)
        console.print(front2, t2)


def highlight_targets(seq: str, targets: list[tuple[int, int]], style="bold magenta"):
    """Highlight the target sequences."""
    text = Text(seq)
    console = Console()
    for start, end in targets:
        text.stylize(style, start, end)
    console.print(text)


def highlight_target(seq: str, start: int, end: int, style="bold magenta"):
    """Highlight the target sequence."""
    text = Text(seq)
    console = Console()
    text.stylize(style, start, end)
    console.print(text)


def hightlight_predict(seq: str, target_start: int, target_end: int, predict_start: int, predict_end: int):
    """Highlight the predicted sequence."""
    text = Text(seq)
    console = Console()

    text.stylize("#adb0b1", target_start, target_end)
    text.stylize("bold magenta", predict_start, predict_end)

    console.print(text)


def summary_predict(predictions, labels):
    """Summarize the predictions and labels."""
    predictions = np.argmax(predictions, axis=2)
    # Initialize lists to hold the filtered predictions and labels

    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()

    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()

    true_predictions, true_labels = rust_summary_predict(predictions, labels, IGNORE_INDEX)
    return true_predictions, true_labels


class LabelHighlighter(RegexHighlighter):
    """Apply style to anything that looks like an email."""

    base_style = "label."
    highlights: typing.ClassVar = [r"(?P<label>1+)"]


def alignment_predict(prediction, label):
    """Print the alignment of the predicted and labeled sequences."""
    import textwrap

    prediction_str = "".join(str(x) for x in prediction)
    label_str = "".join(str(x) for x in label)

    front2 = "L:"
    front1 = "P:"
    theme = Theme({"label.label": "bold magenta"})
    console = Console(highlighter=LabelHighlighter(), theme=theme)
    for l1, l2 in zip(textwrap.wrap(prediction_str), textwrap.wrap(label_str), strict=True):
        ss = f"{front1}{l1}\n{front2}{l2}"
        console.print(ss)
