use std::ops::Range;

use anyhow::{Error, Result};
use bstr::BStr;
use noodles::fastq;
use noodles::fastq::record::Record as FastqRecord;
use rayon::prelude::*;

use crate::{error::EncodingError, fq_encode::RecordData};

use clap::ValueEnum;
use std::fmt::Display;
use std::str::FromStr;

#[derive(Debug, PartialEq, Clone, ValueEnum)]
pub enum ChopType {
    Terminal,
    Internal,
    All,
}

impl Display for ChopType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ChopType::Terminal => write!(f, "terminal"),
            ChopType::Internal => write!(f, "internal"),
            ChopType::All => write!(f, "all"),
        }
    }
}

impl FromStr for ChopType {
    type Err = anyhow::Error;
    fn from_str(s: &str) -> Result<Self> {
        match s {
            "terminal" => Ok(ChopType::Terminal),
            "internal" => Ok(ChopType::Internal),
            "all" => Ok(ChopType::All),
            _ => Err(anyhow::anyhow!("Invalid chop type")),
        }
    }
}

impl ChopType {
    pub fn is_terminal(&self) -> bool {
        matches!(self, ChopType::Terminal)
    }

    pub fn is_internal(&self) -> bool {
        matches!(self, ChopType::Internal)
    }

    pub fn is_all(&self) -> bool {
        matches!(self, ChopType::All)
    }
}

fn _split_records_by_remove_internal<'a>(
    seq: &'a BStr,
    id: &'a BStr,
    qual: &'a [u8],
    target: &'a [Range<usize>],
    min_retain_interval_length: Option<usize>,
) -> Result<(usize, Vec<String>, Vec<&'a BStr>, Vec<&'a BStr>)> {
    let mut seqs = Vec::new();
    let mut quals = Vec::new();
    let mut selected_intervals = Vec::new();

    rayon::scope(|s| {
        s.spawn(|_| {
            let result = remove_intervals_and_keep_left(seq, target).unwrap();
            seqs = result.0;
            selected_intervals = result.1;
        });
        s.spawn(|_| {
            let result = remove_intervals_and_keep_left(qual, target).unwrap();
            quals = result.0;
        });
    });

    // Ensure seqs and quals have the same length; otherwise, return an error or handle as needed
    if seqs.len() != quals.len() {
        return Err(Error::new(
            EncodingError::NotSameLengthForQualityAndSequence(format!(
                "{} seqs: {:?}, quals: {:?}",
                id,
                seqs.len(),
                quals.len()
            )),
        ));
    }

    // ensure the seq and qual length are the same
    for (seq, qual) in seqs.iter().zip(quals.iter()) {
        if seq.len() != qual.len() {
            return Err(Error::new(
                EncodingError::NotSameLengthForQualityAndSequence(format!(
                    "{} seq length {} not equal to qual length {}",
                    id,
                    seq.len(),
                    qual.len(),
                )),
            ));
        }
    }

    let ids: Vec<String> = (0..seqs.len())
        .into_par_iter()
        .map(|x| {
            format!(
                "{}|{}:{}",
                id, selected_intervals[x].start, selected_intervals[x].end,
            )
        })
        .collect();

    let record_count_before_filter = seqs.len();

    if let Some(min_retain_interval_length) = min_retain_interval_length {
        let (filter_ids, (filter_seqs, filter_quals)) = ids
            .into_iter()
            .zip(seqs.into_iter().zip(quals))
            .filter(|(_id, (seq, _qual))| seq.len() >= min_retain_interval_length)
            .unzip();
        return Ok((
            record_count_before_filter,
            filter_ids,
            filter_seqs,
            filter_quals,
        ));
    }

    Ok((record_count_before_filter, ids, seqs, quals))
}

pub fn split_noodle_records_by_intervals(
    seq: &BStr,
    id: &BStr,
    qual: &[u8],
    target: &[Range<usize>],
) -> Result<Vec<FastqRecord>> {
    // get seq by target intervals
    let seqs = target
        .par_iter()
        .map(|interval| seq.get(interval.clone()).unwrap())
        .collect::<Vec<_>>();
    let quals = target
        .par_iter()
        .map(|interval| qual.get(interval.clone()).unwrap())
        .collect::<Vec<_>>();
    let ids = target
        .par_iter()
        .map(|interval| format!("{}|{}:{}", id, interval.start, interval.end))
        .collect::<Vec<_>>();

    Ok(ids
        .into_par_iter()
        .zip(seqs.into_par_iter().zip(quals.into_par_iter()))
        .map(|(id, (seq, qual))| {
            FastqRecord::new(
                fastq::record::Definition::new(id, ""),
                seq.to_vec(),
                qual.to_vec(),
            )
        })
        .collect())
}

pub fn split_noodle_records_by_remove_intervals(
    seq: &BStr,
    id: &BStr,
    qual: &[u8],
    target: &[Range<usize>],
    min_chop_read_length: usize,
    id_annotation: bool,
    chop_type: &ChopType,
) -> Result<Vec<FastqRecord>> {
    let (record_count_before_filter, ids, seqs, quals) =
        _split_records_by_remove_internal(seq, id, qual, target, Some(min_chop_read_length))?;

    let ids_length = record_count_before_filter;

    let current_chop_type = if ids_length == 1 {
        ChopType::Terminal
    } else {
        ChopType::Internal
    };

    if (chop_type.is_terminal() && current_chop_type.is_internal())
        || (chop_type.is_internal() && current_chop_type.is_terminal())
        || (!seqs.is_empty() && seqs[0].len() == seq.len())
    {
        let record = FastqRecord::new(
            fastq::record::Definition::new(id.to_vec(), ""),
            seq.to_vec(),
            qual.to_vec(),
        );
        return Ok(vec![record]);
    }

    let records = ids
        .into_par_iter()
        .zip(seqs.into_par_iter().zip(quals.into_par_iter()))
        .map(|(rid, (rseq, rqual))| {
            let id_str = if id_annotation {
                if current_chop_type == ChopType::Terminal {
                    format!("{}|{}", rid, 'T')
                } else {
                    format!("{}|{}", rid, 'I')
                }
            } else {
                rid
            };

            FastqRecord::new(
                fastq::record::Definition::new(id_str, ""),
                rseq.to_vec(),
                rqual.to_vec(),
            )
        })
        .collect();

    Ok(records)
}

pub fn split_records_by_remove_interval(
    seq: &BStr,
    id: &BStr,
    qual: &[u8],
    target: &[Range<usize>],
    min_chop_read_length: usize,
    id_annotation: bool,
) -> Result<Vec<RecordData>> {
    let (record_count_before_filter, ids, seqs, quals) =
        _split_records_by_remove_internal(seq, id, qual, target, Some(min_chop_read_length))?;
    let ids_length = record_count_before_filter;

    let records = ids
        .into_par_iter()
        .zip(seqs.into_par_iter().zip(quals.into_par_iter()))
        .map(|(id, (seq, qual))| {
            let id_str = if id_annotation {
                if ids_length == 1 {
                    format!("{}|{}", id, 'T')
                } else {
                    format!("{}|{}", id, 'I')
                }
            } else {
                id
            };
            RecordData::new(id_str.into(), seq.into(), qual.into())
        })
        .collect();

    Ok(records)
}

pub fn generate_unmaped_intervals(
    input: &[Range<usize>],
    total_length: usize,
) -> Vec<Range<usize>> {
    // Assuming the input ranges are sorted and non-overlapping
    let mut result = Vec::new();

    if input.is_empty() {
        result.push(0..total_length);
        return result;
    }

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

    // Optionally handle the case after the last interval if necessary,
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
) -> Result<(Vec<&'a BStr>, Vec<Range<usize>>)> {
    let mut intervals = intervals.to_vec();
    intervals.par_sort_by(|a: &Range<usize>, b: &Range<usize>| a.start.cmp(&b.start));
    let selected_intervals = generate_unmaped_intervals(&intervals, seq.len());

    let selected_seq = selected_intervals
        .par_iter()
        .map(|interval| {
            // Check if the interval is valid and starts after the current start point
            if interval.start < seq.len() {
                // Add the segment before the current interval
                Ok(seq[interval.start..interval.end].as_ref())
            } else {
                Err(Error::new(EncodingError::InvalidInterval(format!(
                    "{:?}",
                    interval
                ))))
            }
        })
        .collect::<Result<Vec<_>>>()?;

    Ok((selected_seq, selected_intervals))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_remove_intervals_and_keep_left() {
        let seq = b"abcdefghijklmnopqrstuvwxyz";
        // |a| bcde |fghij| klmno |pqrst| uvwxyz

        let intervals = vec![1..5, 10..15, 20..25];
        let (seqs, _inters) = remove_intervals_and_keep_left(seq, &intervals).unwrap();
        assert_eq!(seqs, vec!["a", "fghij", "pqrst"]);

        let seq = b"abcdefghijklmnopqrstuvwxyz";
        let intervals = vec![5..10, 15..20];
        let (seqs, _inters) = remove_intervals_and_keep_left(seq, &intervals).unwrap();
        assert_eq!(seqs, vec!["abcde", "klmno", "uvwxy"]);

        let seq = b"abcdefghijklmnopqrstuvwxyz";
        let intervals = Vec::new();

        let (seqs, _inters) = remove_intervals_and_keep_left(seq, &intervals).unwrap();
        assert_eq!(seqs[0].to_vec(), seq);
    }

    #[test]
    fn test_generate_unmaped_intervals() {
        let intervals = vec![8100..8123];
        let seq_len = 32768;
        let selected_intervals = generate_unmaped_intervals(&intervals, seq_len);
        assert_eq!(selected_intervals, vec![0..8100, 8123..32767]);
    }
}
