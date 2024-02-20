use std::{
    fmt::Display,
    ops::Range,
    path::{Path, PathBuf},
};

use anyhow::{anyhow, Context, Result};
use derive_builder::Builder;
use log::info;
use pyo3::prelude::*;
use rayon::prelude::*;
use serde_json::json;

use crate::{kmer::to_kmer_target_region, types::Element};

use super::{triat::Encoder, FqEncoderOption};

#[pyclass]
#[derive(Debug, Builder, Default, Clone)]
pub struct JsonEncoder {
    pub option: FqEncoderOption,
}

impl JsonEncoder {
    pub fn new(option: FqEncoderOption) -> Self {
        Self { option }
    }
}

impl Display for JsonEncoder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "FqEncoder {{ option: {} }}", self.option)
    }
}

impl Encoder for JsonEncoder {
    type TargetOutput = Result<Vec<Element>>;
    type EncodeOutput = Result<Vec<serde_json::Value>>;
    type RecordOutput = Result<serde_json::Value>;

    fn encode_target(&self, id: &[u8], kmer_seq_len: Option<usize>) -> Self::TargetOutput {
        let target = Self::parse_target_from_id(id).context("Failed to parse target from ID")?;
        let kmer_target = target
            .par_iter()
            .map(|range| to_kmer_target_region(range, self.option.kmer_size as usize, None))
            .collect::<Result<Vec<Range<usize>>>>()?;

        let encoded_target = if self.option.vectorized_target {
            if kmer_seq_len.is_none() {
                return Err(anyhow!(
                    "kmer_seq_len is None when encodeing target in vector way"
                ));
            }
            let mut encoded_target = vec![0; kmer_seq_len.unwrap()];
            kmer_target
                .iter()
                .for_each(|x| (x.start..x.end).for_each(|i| encoded_target[i] = 1));
            encoded_target
        } else {
            kmer_target
                .into_par_iter()
                .map(|x| [x.start as Element, x.end as Element])
                .flatten()
                .collect()
        };

        Ok(encoded_target)
    }

    fn encode_record(&self, id: &[u8], seq: &[u8], qual: &[u8]) -> Self::RecordOutput {
        // println!("encoding record: {}", String::from_utf8_lossy(id));
        // 1.encode the sequence
        // 2.encode the quality

        // normalize to make sure all the bases are consistently capitalized and
        // that we remove the newlines since this is FASTA
        // change unknwon base to 'N'
        // encode the sequence
        let encoded_seq = self.encoder_seq(seq, self.option.kmer_size, true)?;

        let encoded_seq_str: Vec<String> = encoded_seq
            .into_par_iter()
            .map(|x| String::from_utf8_lossy(x).to_string())
            .collect();

        // encode the quality
        let (encoded_qual, encoded_kmer_qual) =
            self.encode_qual(qual, self.option.kmer_size, self.option.qual_offset);

        let encoded_target = self.encode_target(id, Some(encoded_seq_str.len()))?;

        Ok(json!({
            "id": String::from_utf8_lossy(id),
            "kmer_seq": encoded_seq_str,
            "kmer_qual": encoded_kmer_qual,
            "kmer_target": encoded_target,
            "qual": encoded_qual,
        }))
    }

    fn encode<P: AsRef<Path>>(&mut self, path: P) -> Self::EncodeOutput {
        let records = self.fetch_records(path, self.option.kmer_size)?;
        let data: Vec<serde_json::Value> = records
            .into_par_iter()
            .filter_map(|data| {
                let id = data.id.as_ref();
                let seq = data.seq.as_ref();
                let qual = data.qual.as_ref();

                match self.encode_record(id, seq, qual).context(format!(
                    "encode fq read id {} error",
                    String::from_utf8_lossy(id)
                )) {
                    Ok(result) => Some(result),
                    Err(_e) => None,
                }
            })
            .collect();

        info!("encoded records: {}", data.len());
        Ok(data)
    }

    fn encode_multiple(&mut self, paths: &[PathBuf], parallel: bool) -> Self::EncodeOutput {
        let result = if parallel {
            paths
                .into_par_iter()
                .map(|path| {
                    let mut encoder = self.clone();
                    encoder.encode(path)
                })
                .collect::<Result<Vec<_>>>()?
        } else {
            paths
                .iter()
                .map(|path| {
                    let mut encoder = self.clone();
                    encoder.encode(path)
                })
                .collect::<Result<Vec<_>>>()?
        };

        Ok(result.into_iter().flatten().collect())
    }
}

#[cfg(test)]
mod tests {
    use crate::{fq_encode::FqEncoderOptionBuilder, output::write_json};

    use super::*;

    #[test]
    fn test_encode_fq_for_json_for_large_size_fq() {
        let option = FqEncoderOptionBuilder::default()
            .kmer_size(3)
            .vectorized_target(true)
            .build()
            .unwrap();

        let mut encoder = JsonEncoderBuilder::default()
            .option(option)
            .build()
            .unwrap();

        let result = encoder.encode("tests/data/one_record.fq").unwrap();
        let temp_file = tempfile::NamedTempFile::new().unwrap();
        write_json(temp_file, result).unwrap();
    }
}
