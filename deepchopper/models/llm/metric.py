import evaluate
import numpy as np

clf_metrics = evaluate.combine(["accuracy", "f1", "precision", "recall"])

IGNORE_INDEX = -100


def compute_metrics(p):
    """Compute metrics for a given set of predictions and labels.

    Parameters:
    p (tuple): A tuple containing two numpy arrays - predictions and labels.
               predictions: 3D numpy array of shape (batch_size, sequence_length, num_classes)
               labels: 2D numpy array of shape (batch_size, sequence_length)

    Returns:
    dict: A dictionary containing computed metrics for the predictions.

    Raises:
    ValueError: If the input arrays are not of the expected shape.
    """
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)
    # Initialize lists to hold the filtered predictions and labels
    true_predictions = []
    true_labels = []

    # Filter out '-100' labels and correspondingly filter predictions
    for prediction, label in zip(predictions, labels, strict=True):
        filtered_prediction = []
        filtered_label = []

        for p, l in zip(prediction, label, strict=True):
            if l != IGNORE_INDEX:
                filtered_prediction.append(p)
                filtered_label.append(l)
        true_predictions.append(filtered_prediction)
        true_labels.append(filtered_label)

    for preds, refs in zip(true_predictions, true_labels, strict=True):
        clf_metrics.add_batch(predictions=preds, references=refs)

    return clf_metrics.compute()
