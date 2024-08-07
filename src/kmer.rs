use itertools::Itertools;
use rayon::prelude::*;
use std::ops::Range;

use crate::{
    error::EncodingError,
    types::{Element, Id2KmerTable, Kmer2IdTable},
};
use anyhow::anyhow;
use anyhow::Error;
use anyhow::Result;
use pyo3::prelude::*;

pub fn splite_qual_by_offsets(target: &[usize], offsets: &[(usize, usize)]) -> Result<Vec<usize>> {
    let res: Vec<usize> = offsets
        .par_iter()
        .map(|(start, end)| {
            if start == end {
                // Special token
                0
            } else {
                (*start..*end).map(|i| target[i]).sum::<usize>() / (end - start)
            }
        })
        .collect();
    Ok(res)
}

#[pyfunction]
pub fn vertorize_target(start: usize, end: usize, length: usize) -> Result<Vec<usize>> {
    if start > end || end > length {
        return Err(Error::from(EncodingError::TargetRegionInvalid));
    }

    let mut result = vec![0; length];
    result
        .par_iter_mut()
        .take(end)
        .skip(start)
        .for_each(|x| *x = 1);
    Ok(result)
}

pub fn kmerids_to_seq(kmer_ids: &[Element], id2kmer_table: Id2KmerTable) -> Result<Vec<u8>> {
    let result = kmer_ids
        .par_iter()
        .map(|&id| {
            id2kmer_table
                .get(&id)
                .ok_or(Error::new(EncodingError::InvalidKmerId))
                .map(|kmer| kmer.as_ref())
        })
        .collect::<Result<Vec<_>>>()?;

    kmers_to_seq(result)
}

pub fn to_original_targtet_region(kmer_target: &Range<usize>, k: usize) -> Range<usize> {
    // The start of the target region remains the same
    let original_start = kmer_target.start;

    // Attempt to reverse the end adjustment by adding k - 1, assuming the adjustment was due to k-mer calculation
    let original_end = if kmer_target.end > original_start {
        kmer_target.end + k - 1
    } else {
        kmer_target.end
    };

    original_start..original_end
}

pub fn to_kmer_target_region(
    original_target: &Range<usize>,
    k: usize,
    seq_len: Option<usize>,
) -> Result<Range<usize>> {
    if original_target.start >= original_target.end || k == 0 {
        return Err(Error::from(EncodingError::TargetRegionInvalid));
    }

    if let Some(seq_len) = seq_len {
        // Ensure the target region is valid.
        if original_target.end > seq_len {
            return Err(Error::new(EncodingError::TargetRegionInvalid));
        }
    }

    // Calculate how many k-mers can be formed starting within the original target region.
    let num_kmers_in_target = if original_target.end - original_target.start >= k {
        original_target.end - original_target.start - k + 1
    } else {
        0
    };

    // The new target region starts at the same position as the original target region.
    let new_start = original_target.start;

    // The end of the new target region needs to be adjusted based on the number of k-mers.
    // It is the start position of the last k-mer that can be formed within the original target region.
    let new_end = if num_kmers_in_target > 0 {
        new_start + num_kmers_in_target
    } else {
        original_target.end
    };

    Ok(new_start..new_end)
}

pub fn seq_to_kmers(seq: &[u8], k: usize, overlap: bool) -> Vec<&[u8]> {
    if overlap {
        seq.par_windows(k).collect()
    } else {
        seq.par_chunks(k).collect()
    }
}

pub fn kmers_to_seq(kmers: Vec<&[u8]>) -> Result<Vec<u8>> {
    if kmers.is_empty() {
        return Ok(Vec::new());
    }
    let mut res = kmers[0].to_vec();
    // Iterate over the k-mers, starting from the second one
    let reset: Result<Vec<u8>> = kmers
        .into_par_iter()
        .skip(1)
        .map(|kmer| {
            // Assuming the k-mers are correctly ordered and overlap by k-1,
            // append only the last character of each subsequent k-mer to the sequence.
            kmer.last().ok_or(anyhow!("Invalid kmer")).copied()
        })
        .collect();

    let reset = reset?;

    res.extend(reset);
    Ok(res)
}

#[allow(clippy::type_complexity)]
pub fn seq_to_kmers_and_offset(
    seq: &[u8],
    kmer_size: usize,
    overlap: bool,
) -> Result<Vec<(&[u8], (usize, usize))>> {
    // Check for invalid kmer_size
    if kmer_size == 0 || kmer_size > seq.len() {
        return Err(EncodingError::SeqShorterThanKmer.into());
    }

    if seq.is_empty() {
        return Ok(Vec::new());
    }

    if overlap {
        // Overlapping case: use .windows() with step of 1 (default behavior of .windows())
        Ok(seq
            .par_windows(kmer_size)
            .enumerate()
            .map(|(i, kmer)| (kmer, (i, i + kmer_size)))
            .collect())
    } else {
        // Non-overlapping case: iterate with steps of kmer_size
        Ok(seq
            .par_chunks(kmer_size)
            .enumerate()
            .filter_map(|(i, chunk)| {
                if chunk.len() == kmer_size {
                    Some((chunk, (i * kmer_size, i * kmer_size + kmer_size)))
                } else {
                    // ignore the last chunk if it's shorter than kmer_size
                    None
                }
            })
            .collect())
    }
}

pub fn generate_kmers_table(base: &[u8], k: u8) -> Kmer2IdTable {
    generate_kmers(base, k)
        .into_par_iter()
        .enumerate()
        .map(|(id, kmer)| (kmer, id as Element))
        .collect()
}

pub fn generate_kmers(bases: &[u8], k: u8) -> Vec<Vec<u8>> {
    // Convert u8 slice to char Vec directly where needed
    (0..k)
        .map(|_| bases.iter().map(|&c| c as char)) // Direct conversion to char iter
        .multi_cartesian_product()
        .map(|combo| combo.into_iter().map(|c| c as u8).collect::<Vec<_>>())
        .collect::<Vec<_>>()
}

#[cfg(test)]
mod tests {
    use super::*;
    use bio::utils::Interval;

    #[test]
    fn test_seq_to_kmers() {
        let seq1 = b"ATCGT";
        let k = 2;
        let kmers = seq_to_kmers(seq1, k, true);
        assert_eq!(kmers.len(), seq1.len() - k + 1);

        let seq2 = b"AT";
        let k = 3;
        let kmers = seq_to_kmers(seq2, k, true);
        println!("{:?}", kmers);
        assert_eq!(kmers.len(), 0);
    }

    #[test]
    fn test_generate_kmers() {
        // Test case 1: bases = ['A', 'C', 'G', 'T'], k = 2
        let bases1 = b"ACGT";
        let k1 = 2;
        let expected1 = vec![
            "AA", "AC", "AG", "AT", "CA", "CC", "CG", "CT", "GA", "GC", "GG", "GT", "TA", "TC",
            "TG", "TT",
        ]
        .into_iter()
        .map(|s| s.chars().map(|c| c as u8).collect::<Vec<_>>())
        .collect::<Vec<_>>();

        assert_eq!(generate_kmers(bases1, k1), expected1);

        // Test case 2: bases = ['A', 'C'], k = 3
        let bases2 = b"AC";
        let k2 = 3;
        let expected2 = vec!["AAA", "AAC", "ACA", "ACC", "CAA", "CAC", "CCA", "CCC"]
            .into_iter()
            .map(|s| s.chars().map(|c| c as u8).collect::<Vec<_>>())
            .collect::<Vec<_>>();
        assert_eq!(generate_kmers(bases2, k2), expected2);
    }

    #[test]
    fn test_generate_kmers_table() {
        let base = b"ACGT";
        let k = 2;
        let expected_table: Kmer2IdTable = [
            ("AA", 0),
            ("GC", 9),
            ("GT", 11),
            ("CA", 4),
            ("TA", 12),
            ("TC", 13),
            ("CG", 6),
            ("CT", 7),
            ("GA", 8),
            ("AG", 2),
            ("AC", 1),
            ("AT", 3),
            ("CC", 5),
            ("GG", 10),
            ("TG", 14),
            ("TT", 15),
        ]
        .iter()
        .map(|&(kmer, id)| (kmer.chars().map(|c| c as u8).collect(), id))
        .collect();

        assert_eq!(generate_kmers_table(base, k), expected_table);
    }

    #[test]
    fn test_generate_kmers_table_empty_base() {
        use ahash::HashMap;
        use ahash::HashMapExt;

        let base = b"";
        let k = 2;
        let expected_table: Kmer2IdTable = HashMap::new();
        assert_eq!(generate_kmers_table(base, k), expected_table);
    }

    #[test]
    fn test_construct_seq_from_kmers() {
        let k = 3;
        let seq = b"AAACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT";
        let kmers = seq_to_kmers(seq, k, true);
        let kmers_as_bytes: Vec<&[u8]> = kmers.into_iter().collect();
        let result = kmers_to_seq(kmers_as_bytes).unwrap();
        assert_eq!(seq.to_vec(), result);
    }

    #[test]
    fn test_update_target_region() {
        let original_target: Interval<usize> = (2..6).into(); // Target region [2, 6)
        let k = 3; // K-mer size
        let new_target_region = to_kmer_target_region(&original_target, k, None).unwrap();
        assert_eq!(new_target_region, 2..4);
    }

    #[test]
    fn test_update_target_region_valid() {
        let original_target = Interval::new(0..10).unwrap();
        let k = 3;
        let seq_len = Some(20);

        let result = to_kmer_target_region(&original_target, k, seq_len);

        assert!(result.is_ok());
        let new_target = result.unwrap();

        assert_eq!(new_target.start, original_target.start);
        assert_eq!(new_target.end, original_target.start + 8);
    }

    #[test]
    fn test_update_target_region_invalid_start_greater_than_end() {
        let original_target = Interval::new(10..10).unwrap();
        let k = 3;
        let seq_len = Some(20);

        let result = to_kmer_target_region(&original_target, k, seq_len);
        assert!(result.is_err());

        assert_eq!(
            result.unwrap_err().to_string(),
            EncodingError::TargetRegionInvalid.to_string()
        );
    }

    #[test]
    fn test_update_target_region_invalid_end_greater_than_seq_len() {
        let original_target = Interval::new(0..25).unwrap();
        let k = 3;
        let seq_len = Some(20);

        let result = to_kmer_target_region(&original_target, k, seq_len);

        assert!(result.is_err());
        assert_eq!(
            result.unwrap_err().to_string(),
            EncodingError::TargetRegionInvalid.to_string()
        );
    }

    #[test]
    fn test_to_original_target_region() {
        // Test case 1: kmer_target.end > original_start
        let kmer_target = 2..5;
        let k = 3;
        let expected = 2..7;

        assert_eq!(
            to_kmer_target_region(&expected, k, None).unwrap(),
            kmer_target
        );
        assert_eq!(to_original_targtet_region(&kmer_target, k), expected);

        // Test case 3: kmer_target.end == original_start
        let kmer_target = 5..5;
        let k = 3;
        let expected = 5..5;
        assert_eq!(to_original_targtet_region(&kmer_target, k), expected);
    }

    #[test]
    fn test_seq_to_kmers_and_offset_overlap() {
        let seq = b"ATCGATCGATCG";
        let kmer_size = 4;
        let overlap = true;
        let result = seq_to_kmers_and_offset(seq, kmer_size, overlap).unwrap();
        assert_eq!(result.len(), seq.len() - kmer_size + 1);
        assert_eq!(result[0], (&b"ATCG"[..], (0, 4)));
        assert_eq!(result[1], (&b"TCGA"[..], (1, 5)));
        assert_eq!(result[result.len() - 1], (&b"ATCG"[..], (8, 12)));
    }

    #[test]
    fn test_seq_to_kmers_and_offset_non_overlap() {
        let seq = b"ATCGATCGATCG";
        let kmer_size = 4;
        let overlap = false;
        let result = seq_to_kmers_and_offset(seq, kmer_size, overlap).unwrap();
        assert_eq!(result.len(), seq.len() / kmer_size);
        assert_eq!(result[0], (&b"ATCG"[..], (0, 4)));
        assert_eq!(result[1], (&b"ATCG"[..], (4, 8)));
    }

    #[test]
    fn test_vertorize_target_valid() {
        let start = 3;
        let end = 5;
        let result = vertorize_target(start, end, 6).unwrap();
        assert_eq!(result, vec![0, 0, 0, 1, 1, 0]);

        let rr = vertorize_target(0, 0, 6).unwrap();
        assert_eq!(rr, vec![0, 0, 0, 0, 0, 0]);
    }

    #[test]
    fn test_vertorize_target_invalid() {
        let start = 5;
        let end = 0;
        let result = vertorize_target(start, end, 2);
        assert!(result.is_err());
    }
}
