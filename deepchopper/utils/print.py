from rich.console import Console
from rich.text import Text


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
            if l != -100:
                filtered_prediction.append(p)
                filtered_label.append(l)
        true_predictions.append(filtered_prediction)
        true_labels.append(filtered_label)

    return true_predictions, true_labels


def alignment_predict(prediction, label):
    import textwrap

    prediction_str = "".join(str(x) for x in prediction)
    label_str = "".join(str(x) for x in label)
    for _l1, _l2 in zip(textwrap.wrap(prediction_str), textwrap.wrap(label_str), strict=False):
        pass
