use anyhow::Result;
use bstr::BStr;
use rayon::prelude::*;
use std::ops::Range;

use crate::{error::EncodingError, fq_encode::RecordData};

pub fn generate_records_by_remove_interval(
    seq: &BStr,
    id: &BStr,
    qual: &BStr,
    target: &[Range<usize>],
) -> Result<Vec<RecordData>, EncodingError> {
    let mut seqs = Vec::new();
    let mut quals = Vec::new();

    rayon::scope(|s| {
        s.spawn(|_| {
            seqs = remove_intervals_and_keep_left(seq, target).unwrap();
        });
        s.spawn(|_| {
            quals = remove_intervals_and_keep_left(qual, target).unwrap();
        });
    });

    // Ensure seqs and quals have the same length; otherwise, return an error or handle as needed
    if seqs.len() != qual.len() {
        return Err(EncodingError::NotSameLengthForQualityAndSequence(format!(
            "seqs: {:?}, quals: {:?}",
            seqs.len(),
            quals.len()
        )));
    }

    let ids: Vec<String> = (0..seqs.len()).map(|x| format!("{}{}", id, x)).collect();

    let records = ids
        .into_iter()
        .zip(seqs.into_iter().zip(quals))
        .map(|(id, (seq, qual))| {
            RecordData::new(id.as_bytes().to_vec(), seq.to_vec(), qual.to_vec())
        })
        .collect();

    Ok(records)
}

pub fn generate_unmaped_intervals(
    input: &[Range<usize>],
    total_length: usize,
) -> Vec<Range<usize>> {
    if input.is_empty() {
        return vec![];
    }

    // Assuming the input ranges are sorted and non-overlapping
    let mut result = Vec::new();

    // Initial start for the very first interval
    let mut current_start = 0;

    for range in input.iter() {
        // Check if there's a gap between current_start and the start of the current range
        if current_start < range.start {
            result.push(current_start..range.start);
        }
        // Update current_start to the end of the current range
        current_start = range.end;
    }

    // Optionally handle the case after the last interval if necessary
    // For example, if you know the total length and want to add an interval up to that length

    if current_start < total_length - 1 {
        result.push(current_start..total_length - 1);
    }

    result
}

// Function to remove intervals from a sequence and keep the remaining parts
pub fn remove_intervals_and_keep_left<'a>(
    seq: &'a [u8],
    intervals: &[Range<usize>],
) -> Result<Vec<&'a BStr>, EncodingError> {
    let mut intervals = intervals.to_vec();
    intervals.par_sort_by(|a: &Range<usize>, b: &Range<usize>| a.start.cmp(&b.start));
    let slected_intervals = generate_unmaped_intervals(&intervals, seq.len());

    slected_intervals
        .par_iter()
        .map(|interval| {
            // Check if the interval is valid and starts after the current start point
            if interval.start < seq.len() {
                // Add the segment before the current interval
                let res = seq[interval.start..interval.end].as_ref();
                Ok(res)
            } else {
                Err(EncodingError::InvalidInterval(format!("{:?}", interval)))
            }
        })
        .collect::<Result<Vec<_>, EncodingError>>()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_remove_intervals_and_keep_left() {
        let seq = b"abcdefghijklmnopqrstuvwxyz";
        // |a| bcde |fghij| klmno |pqrst| uvwxyz

        let intervals = vec![1..5, 10..15, 20..25];
        let result = remove_intervals_and_keep_left(seq, &intervals).unwrap();
        assert_eq!(result, vec!["a", "fghij", "pqrst"]);

        let seq = b"abcdefghijklmnopqrstuvwxyz";
        let intervals = vec![5..10, 15..20];
        let result = remove_intervals_and_keep_left(seq, &intervals).unwrap();
        assert_eq!(result, vec!["abcde", "klmno", "uvwxy"]);
    }
}