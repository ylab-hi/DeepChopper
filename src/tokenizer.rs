use pyo3::prelude::*;
use tokenizers::tokenizer::NormalizedString;
use tokenizers::tokenizer::Split;
use tokenizers::tokenizer::{PreTokenizedString, PreTokenizer, Result};

#[pyclass]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct KmerOverlapPreTokenizer {
    pub k: usize,
}

#[pymethods]
impl KmerOverlapPreTokenizer {
    #[new]
    pub fn new(k: usize) -> Self {
        Self { k }
    }
}

impl PreTokenizer for KmerOverlapPreTokenizer {
    fn pre_tokenize(&self, pretokenized: &mut PreTokenizedString) -> Result<()> {
        pretokenized.split(|_, normalized: NormalizedString| {
            let seq = normalized.get();
            let mut splits = Vec::new();

            if seq.len() >= self.k {
                for start in 0..=seq.len() - self.k {
                    let end = start + self.k;
                    // Extract the substring for the current k-mer
                    let kmer = &seq[start..end];
                    // Create a new NormalizedString for this k-mer
                    let kmer_normalized = NormalizedString::from(kmer);

                    // Create a Split for this k-mer. Since we cannot directly set offsets in the Split,
                    // this assumes that the conceptual offset handling aligns with the k-mer's position.
                    let split = Split::from(kmer_normalized);
                    splits.push(split);
                }
            } else {
                // Handle case where the entire sequence is shorter than k
                let split = Split::from(normalized);
                splits.push(split);
            }
            Ok(splits.into_iter())
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokenizers::tokenizer::{Result, Tokenizer};

    #[test]
    fn test_tokenizer() {
        let tokenizer = Tokenizer::from_pretrained("bert-base-cased", None).unwrap();

        let encoding = tokenizer.encode("Hey there!", false).unwrap();
        println!("{:?}", encoding.get_tokens());
    }

    #[test]
    fn test_pre() {
        let mut pretokenized = tokenizers::tokenizer::PreTokenizedString::from("ATCGG");
        let t = KmerOverlapPreTokenizer::new(3);
        let pre_tokenized = t.pre_tokenize(&mut pretokenized);
        println!("{:?}", pre_tokenized);

        let res: Vec<(String, (usize, usize))> = pretokenized
            .get_splits(
                tokenizers::OffsetReferential::Original,
                tokenizers::OffsetType::Char,
            )
            .into_iter()
            .map(|(s, o, _)| (s.to_owned(), o))
            .collect();
        println!("{:?}", res);
    }
}
