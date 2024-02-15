use itertools::Itertools;
use needletail::kmer::Kmers;
use rayon::prelude::*;
use std::{collections::HashMap, ops::Range};

use crate::{error::EncodingError, fq_encode::Element};
use anyhow::Result;

pub type KmerTable = HashMap<Vec<u8>, Element>;

fn reverse_update_target_region(updated_target: &Range<usize>, k: usize) -> Range<usize> {
    // The start of the target region remains the same
    let original_start = updated_target.start;

    // Attempt to reverse the end adjustment by adding k - 1, assuming the adjustment was due to k-mer calculation
    let original_end = if updated_target.end > original_start {
        updated_target.end + k - 1
    } else {
        updated_target.end
    };

    original_start..original_end
}

pub fn to_kmer_target_region(
    original_target: &Range<usize>,
    k: usize,
    seq_len: Option<usize>,
) -> Result<Range<usize>> {
    if original_target.start >= original_target.end || k == 0 {
        return Err(EncodingError::TargetRegionInvalid.into());
    }

    if let Some(seq_len) = seq_len {
        // Ensure the target region is valid.
        if original_target.end > seq_len {
            return Err(EncodingError::TargetRegionInvalid.into());
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

pub fn seq_to_kmers(seq: &[u8], k: u8) -> Kmers {
    Kmers::new(seq, k)
}

pub fn kmers_to_seq(kmers: Vec<&[u8]>) -> Vec<u8> {
    if kmers.is_empty() {
        return Vec::new();
    }
    // Initialize the sequence with the first k-mer
    let mut sequence = kmers[0].to_vec();
    // Iterate over the k-mers, starting from the second one
    for kmer in kmers.into_iter().skip(1) {
        // Assuming the k-mers are correctly ordered and overlap by k-1,
        // append only the last character of each subsequent k-mer to the sequence.
        if let Some(&last_char) = kmer.last() {
            sequence.push(last_char);
        }
    }

    sequence
}

pub fn generate_kmers_table(base: &[u8], k: u8) -> KmerTable {
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
        let kmers = seq_to_kmers(seq1, k);
        assert_eq!(kmers.into_iter().count(), seq1.len() - k as usize + 1);

        let seq2 = b"AT";
        let k = 3;
        let kmers = seq_to_kmers(seq2, k);
        assert_eq!(kmers.into_iter().count(), 0);
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
        let expected_table: KmerTable = [
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
        let base = b"";
        let k = 2;
        let expected_table: KmerTable = HashMap::new();
        assert_eq!(generate_kmers_table(base, k), expected_table);
    }

    #[test]
    fn test_construct_seq_from_kmers() {
        let k = 3;
        let seq = b"AAACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT";
        let kmers = seq_to_kmers(seq, k);
        let kmers_as_bytes: Vec<&[u8]> = kmers.into_iter().collect();
        let result = kmers_to_seq(kmers_as_bytes);
        assert_eq!(seq.to_vec(), result);
    }

    #[test]
    fn test_update_target_region() {
        let original_target: Interval<usize> = (2..6).into(); // Target region [2, 6)
        let k = 3; // K-mer size
        let new_target_region = to_kmer_target_region(&original_target, k, None).unwrap();
        assert_eq!(new_target_region, (2..4));
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
}
