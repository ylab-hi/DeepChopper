use anyhow::Result;
use ndarray::prelude::*;
use noodles::fastq;
use rayon::prelude::*;
use std::{fs::File, io::BufReader, path::Path};

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

pub fn collect_and_split_dataset<P: AsRef<Path>>(
    internal_fq_path: P,
    terminal_fq_path: P,
    negative_fq_path: P,
    total_reads: f32,
    train_ratio: f32, // 0.8
    val_ratio: f32,   // 0.1
    test_ratio: f32,  // 0.1
    iternal_adapter_ratio: f32,
    positive_ratio: f32,
) -> Result<()> {
    if train_ratio + val_ratio + test_ratio != 1.0 {
        return Err(anyhow::anyhow!(
            "train_ratio + val_ratio + test_ratio must be equal to 1.0"
        ));
    }

    let terminal_adapter_ratio = 1.0 - iternal_adapter_ratio;
    let negative_ratio = 1.0 - positive_ratio;

    // calculate the number of reads in each file
    let train_count = total_reads * train_ratio;
    let val_count = total_reads * val_ratio;
    let test_count = total_reads * test_ratio;

    let train_positive_count = train_count * positive_ratio;
    let val_positive_count = val_count * positive_ratio;
    let test_positive_count = test_count * positive_ratio;

    let train_negative_count = (train_count * negative_ratio) as usize;
    let val_negative_count = (val_count * negative_ratio) as usize;
    let test_negative_count = (test_count * negative_ratio) as usize;

    let train_internal_adapter_count = (train_positive_count * iternal_adapter_ratio) as usize;
    let train_terminal_adapter_count = (train_positive_count * terminal_adapter_ratio) as usize;

    let val_internal_adapter_count = (val_positive_count * iternal_adapter_ratio) as usize;
    let val_terminal_adapter_count = (val_positive_count * terminal_adapter_ratio) as usize;

    let test_internal_adapter_count = (test_positive_count * iternal_adapter_ratio) as usize;
    let test_terminal_adapter_count = (test_positive_count * terminal_adapter_ratio) as usize;

    let mut internal_fq_reader = File::open(internal_fq_path.as_ref())
        .map(BufReader::new)
        .map(fastq::Reader::new)?;
    let mut terminal_fq_reader = File::open(terminal_fq_path.as_ref())
        .map(BufReader::new)
        .map(fastq::Reader::new)?;
    let mut negative_fq_reader = File::open(negative_fq_path.as_ref())
        .map(BufReader::new)
        .map(fastq::Reader::new)?;

    let mut train_writer = fastq::Writer::new(File::create("train.fq")?);
    let mut val_writer = fastq::Writer::new(File::create("val.fq")?);
    let mut test_writer = fastq::Writer::new(File::create("test.fq")?);

    // write for positive train records
    internal_fq_reader
        .records()
        .take(train_internal_adapter_count)
        .for_each(|record| {
            train_writer.write_record(&record.unwrap()).unwrap();
        });
    terminal_fq_reader
        .records()
        .take(train_terminal_adapter_count)
        .for_each(|record| {
            train_writer.write_record(&record.unwrap()).unwrap();
        });
    negative_fq_reader
        .records()
        .take(train_negative_count)
        .for_each(|record| {
            train_writer.write_record(&record.unwrap()).unwrap();
        });

    // write for positive val records
    internal_fq_reader
        .records()
        .skip(train_internal_adapter_count)
        .take(val_internal_adapter_count)
        .for_each(|record| {
            val_writer.write_record(&record.unwrap()).unwrap();
        });
    terminal_fq_reader
        .records()
        .skip(train_terminal_adapter_count)
        .take(val_terminal_adapter_count)
        .for_each(|record| {
            val_writer.write_record(&record.unwrap()).unwrap();
        });
    negative_fq_reader
        .records()
        .skip(train_negative_count)
        .take(val_negative_count)
        .for_each(|record| {
            val_writer.write_record(&record.unwrap()).unwrap();
        });

    // write for positive test records
    internal_fq_reader
        .records()
        .skip(train_internal_adapter_count + val_internal_adapter_count)
        .take(test_internal_adapter_count)
        .for_each(|record| {
            test_writer.write_record(&record.unwrap()).unwrap();
        });
    terminal_fq_reader
        .records()
        .skip(train_terminal_adapter_count + val_terminal_adapter_count)
        .take(test_terminal_adapter_count)
        .for_each(|record| {
            test_writer.write_record(&record.unwrap()).unwrap();
        });
    negative_fq_reader
        .records()
        .skip(train_negative_count + val_negative_count)
        .take(test_negative_count)
        .for_each(|record| {
            test_writer.write_record(&record.unwrap()).unwrap();
        });
    Ok(())
}

pub fn smooth_label(labels: &[u8]) -> Vec<u8> {
    let mut smoothed_labels = Vec::with_capacity(labels.len());
    for i in 0..labels.len() {
        let smooth_count = labels
            .iter()
            .skip(i.saturating_sub(1))
            .take(3)
            .filter(|&x| *x == 1)
            .count();
        smoothed_labels.push(if smooth_count >= 2 { 1 } else { 0 });
    }
    smoothed_labels
}

pub fn smooth_label_region(labels: &[u8]) -> Vec<u8> {
    // find all region that has 1 continuous 1s
    // let regions = vec![];
    // labels.iter().map(|&x| x).collect();
    todo!()
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

    #[test]
    fn name() {
        collect_and_split_dataset(
            "tests/data/1000_records.fq",
            "tests/data/1000_records.fq",
            "tests/data/1000_records.fq",
            100.0,
            0.8,
            0.1,
            0.1,
            0.5,
            0.9,
        )
        .unwrap();
    }
}
