import numpy as np
from rich.console import Console
from rich.highlighter import RegexHighlighter
from rich.text import Text
from rich.theme import Theme

from deepchopper.models.hyena import IGNORE_INDEX


def highlight_target(seq: str, start: int, end: int, style="bold magenta"):
    text = Text(seq)
    console = Console()
    text.stylize(style, start, end)
    console.print(text)


def hightlight_predict(
    seq: str, target_start: int, target_end: int, predict_start: int, predict_end: int
):
    text = Text(seq)
    console = Console()

    text.stylize("#adb0b1", target_start, target_end)
    text.stylize("bold magenta", predict_start, predict_end)

    console.print(text)


def summary_predict(predictions, labels):
    predictions = np.argmax(predictions, axis=2)
    # Initialize lists to hold the filtered predictions and labels
    true_predictions = []
    true_labels = []

    # Filter out '-100' labels and correspondingly filter predictions
    for prediction, label in zip(predictions, labels, strict=False):
        filtered_prediction = []
        filtered_label = []

        for p, l in zip(prediction, label, strict=False):
            if l != IGNORE_INDEX:
                filtered_prediction.append(p)
                filtered_label.append(l)
        true_predictions.append(filtered_prediction)
        true_labels.append(filtered_label)

    return true_predictions, true_labels


class LabelHighlighter(RegexHighlighter):
    """Apply style to anything that looks like an email."""

    base_style = "label."
    highlights = [r"(?P<label>1+)"]


def alignment_predict(prediction, label):
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
