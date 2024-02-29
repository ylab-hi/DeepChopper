use ndarray::prelude::*;
use rayon::prelude::*;

pub fn summary_predict_for_array(
    predictions: &Array2<i8>,
    labels: &Array2<i8>,
) -> (Array2<i8>, Array2<i8>) {
    // First, ensure that predictions and labels have the same shape
    assert_eq!(predictions.dim(), labels.dim());

    // Flatten both arrays
    let flat_predictions = predictions.iter().copied().collect::<Vec<i8>>();
    let flat_labels = labels.iter().copied().collect::<Vec<i8>>();

    // Filter predictions and labels where label != -100
    let filtered: Vec<(i8, i8)> = flat_labels
        .into_iter()
        .zip(flat_predictions)
        .filter(|&(l, _)| l != -100)
        .collect();

    // Separate the filtered predictions and labels
    let (filtered_labels, filtered_predictions): (Vec<i8>, Vec<i8>) = filtered.into_iter().unzip();

    let shape = (filtered_labels.len(), 1);

    // Convert back to Array2 - note this will be 1D arrays since original structure is lost
    let filtered_labels_array = Array1::from(filtered_labels).into_shape(shape).unwrap();
    let filtered_predictions_array = Array1::from(filtered_predictions)
        .into_shape(shape)
        .unwrap();

    (filtered_predictions_array, filtered_labels_array)
}

pub fn summary_predict(
    predictions: &[Vec<i8>],
    labels: &[Vec<i8>],
) -> (Vec<Vec<i8>>, Vec<Vec<i8>>) {
    predictions
        .par_iter()
        .zip(labels.par_iter())
        .map(|(prediction, label)| {
            let (filter_predictions, filter_labels): (Vec<i8>, Vec<i8>) = prediction
                .iter()
                .zip(label.iter())
                .fold((vec![], vec![]), |mut acc, (&p, &l)| {
                    if l != -100 {
                        acc.1.push(l);
                        acc.0.push(p);
                    }
                    acc
                });
            (filter_predictions, filter_labels)
        })
        .unzip()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_summary_predict() {
        let predictions = vec![vec![0, 0, 1], vec![1, 1, 1]];
        let labels = vec![vec![0, -100, 1], vec![-100, 1, -100]];
        let (true_predictions, true_labels) = summary_predict(&predictions, &labels);
        let expected_predictions = vec![vec![0, 1], vec![1]];
        let expected_labels = vec![vec![0, 1], vec![1]];
        assert_eq!(true_predictions, expected_predictions);
        assert_eq!(true_labels, expected_labels);
    }
}
