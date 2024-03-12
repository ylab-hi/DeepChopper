use anyhow::Result;
use noodles::fastq;
use rayon::prelude::*;
use std::{fs::File, io::BufReader, ops::Range, path::Path};

pub fn summary_predict(
    predictions: &[Vec<i8>],
    labels: &[Vec<i8>],
    ignore_label: i8,
) -> (Vec<Vec<i8>>, Vec<Vec<i8>>) {
    predictions
        .par_iter()
        .zip(labels.par_iter())
        .map(|(prediction, label)| {
            let (filter_predictions, filter_labels): (Vec<i8>, Vec<i8>) = prediction
                .iter()
                .zip(label.iter())
                .fold((vec![], vec![]), |mut acc, (&p, &l)| {
                    if l != ignore_label {
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

/// find 1s regions in the labels e.g. 00110011100
pub fn get_label_region(labels: &[u8]) -> Vec<Range<usize>> {
    let mut regions = vec![];

    let mut start = 0;
    let mut end = 0;

    for (i, label) in labels.iter().enumerate() {
        if *label == 1 {
            if start == 0 {
                start = i;
            }
            end = i;
        } else if start != 0 {
            regions.push(start..end + 1);
            start = 0;
            end = 0;
        }
    }

    if start != 0 {
        regions.push(start..end + 1);
    }

    regions
}

/// merge 1s regions that are close to each other
/// e.g. 00110011100 -> 0011111100
pub fn smooth_label_region(
    labels: &[u8],
    merge_threshold: usize,
    region_distance_threshold: usize,
) -> Vec<Range<usize>> {
    let labels_region = get_label_region(labels);
    let mut smoothed_regions = vec![];

    for region in labels_region {
        if region.len() < region_distance_threshold {
            continue;
        }

        if smoothed_regions.is_empty() {
            smoothed_regions.push(region);
            continue;
        }

        let last_region = smoothed_regions.last_mut().unwrap();
        if region.start - last_region.end <= merge_threshold {
            last_region.end = region.end;
        } else {
            smoothed_regions.push(region);
        }
    }
    smoothed_regions
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_summary_predict() {
        let predictions = vec![vec![0, 0, 1], vec![1, 1, 1]];
        let labels = vec![vec![0, -100, 1], vec![-100, 1, -100]];
        let (true_predictions, true_labels) = summary_predict(&predictions, &labels, -100);
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

    #[test]
    fn test_get_label_region_empty() {
        let labels = vec![];
        let regions = get_label_region(&labels);
        assert_eq!(regions.len(), 0);
    }

    #[test]
    fn test_get_label_region_no_label() {
        let labels = [0, 0, 0, 0];
        let regions = get_label_region(&labels);
        assert_eq!(regions.len(), 0);
    }

    #[test]
    fn test_get_label_region_single_label() {
        let labels = [0, 1, 0, 0, 0];
        let regions = get_label_region(&labels);
        assert_eq!(regions, vec![1..2]);
    }

    #[test]
    fn test_get_label_region_multiple_labels() {
        let labels = [0, 1, 1, 0, 1, 1, 0];
        let regions = get_label_region(&labels);
        assert_eq!(regions, vec![1..3, 4..6]);
    }

    #[test]
    fn test_get_label_region_label_at_end() {
        let labels = [0, 1, 1, 0, 1, 1];
        let regions = get_label_region(&labels);
        assert_eq!(regions, vec![1..3, 4..6]);
    }

    #[test]
    fn test_smooth_label_region() {
        // Test case 1: no regions to merge
        let labels1 = [0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0]; // 2,4 and 6,9
        let merge_threshold1 = 2;
        let distance_threshold1 = 2;
        let expected_result1 = vec![2..9];
        assert_eq!(
            smooth_label_region(&labels1, merge_threshold1, distance_threshold1),
            expected_result1
        );

        // Test case 2: merge 1s regions that are close to each other
        let labels2 = [0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0]; // 2,4 and 6,9
        let merge_threshold2 = 1;
        let distance_threshold2 = 2;
        let expected_result2 = vec![2..4, 6..9];
        assert_eq!(
            smooth_label_region(&labels2, merge_threshold2, distance_threshold2),
            expected_result2
        );
    }
}
