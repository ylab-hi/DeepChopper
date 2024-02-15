use itertools::Itertools;
use needletail::kmer::Kmers;
use std::collections::HashMap;

use crate::fq_encode::Element;

pub type KmerTable = HashMap<Vec<u8>, Element>;

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
        .into_iter()
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
}
