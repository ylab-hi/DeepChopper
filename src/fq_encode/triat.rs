use anyhow::Result;
use anyhow::{anyhow, Context};
use log::info;
use needletail::Sequence;
use noodles::fastq;
use rayon::prelude::*;
use std::fs::File;
use std::io::BufReader;
use std::ops::Range;
use std::path::{Path, PathBuf};

use crate::types::Element;

use super::RecordData;

pub trait Encoder {
    type TargetOutput;
    type EncodeOutput;
    type RecordOutput;

    fn encode_target(&self, id: &[u8], kmer_seq_len: Option<usize>) -> Self::TargetOutput;

    fn encode_multiple(&mut self, paths: &[PathBuf], parallel: bool) -> Self::EncodeOutput;

    fn encode<P: AsRef<Path>>(&mut self, path: P) -> Self::EncodeOutput;

    fn encode_record(&self, id: &[u8], seq: &[u8], qual: &[u8]) -> Self::RecordOutput;

    fn parse_target_from_id(src: &[u8]) -> Result<Vec<Range<usize>>> {
        // check empty input
        if src.is_empty() {
            return Ok(Vec::new());
        }

        // @462:528,100:120|738735b7-2105-460e-9e56-da980ef816c2+4f605fb4-4107-4827-9aed-9448d02834a8
        // removea content after |
        let number_part = src
            .split(|&c| c == b'|')
            .next()
            .context("Failed to get number part")?;

        number_part
            .par_split(|&c| c == b',')
            .map(|target| {
                let mut parts = target.split(|&c| c == b':');
                let start: usize =
                    lexical::parse(parts.next().ok_or(anyhow!("parse number error"))?)?;
                let end: usize =
                    lexical::parse(parts.next().ok_or(anyhow!("parse number error"))?)?;
                Ok(start..end)
            })
            .collect::<Result<Vec<_>>>()
    }

    fn fetch_records<P: AsRef<Path>>(&mut self, path: P, kmer_size: u8) -> Result<Vec<RecordData>> {
        info!("fetching records from {}", path.as_ref().display());
        let mut reader = File::open(path.as_ref())
            .map(BufReader::new)
            .map(fastq::Reader::new)?;

        let mut records: Vec<RecordData> = Vec::new();
        let mut record = fastq::Record::default();

        while reader.read_record(&mut record)? > 0 {
            let id = record.definition().name();
            let seq = record.sequence();
            let normalized_seq = seq.normalize(false);
            let qual = record.quality_scores();
            let seq_len = normalized_seq.len();
            let qual_len = qual.len();

            if seq_len < kmer_size as usize {
                continue;
            }

            if seq_len != qual_len {
                return Err(anyhow!(
                    "record: id {} seq_len != qual_len",
                    String::from_utf8_lossy(id)
                ));
            }

            records.push((id.to_vec(), seq.to_vec(), qual.to_vec()).into());
        }

        info!("total records: {}", records.len());
        Ok(records)
    }

    fn encode_qual(
        &self,
        qual: &[u8],
        kmer_size: u8,
        qual_offset: u8,
    ) -> (Vec<Element>, Vec<Element>) {
        // input is quality of fastq
        // 1. convert the quality to a score
        // 2. return the score
        let encoded_qual: Vec<u8> = qual
            .par_iter()
            .map(|&q| {
                // Convert ASCII to Phred score for Phred+33 encoding
                q - qual_offset
            })
            .collect();

        let encoded_kmer_qual: Vec<Element> = encoded_qual
            .kmers(kmer_size)
            .par_bridge()
            .map(|q| {
                let values = q.to_vec();
                // get average value of the kmer
                let mean = values.iter().sum::<u8>() / values.len() as u8;
                mean as Element
            })
            .collect();

        (
            encoded_qual.into_par_iter().map(|x| x as Element).collect(),
            encoded_kmer_qual,
        )
    }

    fn encoder_seq<'a>(&self, seq: &'a [u8], kmer_size: u8) -> Vec<&'a [u8]> {
        seq.kmers(kmer_size).collect()
    }
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn test_parse_target_from_id() {
        // Test case 1: Valid input
        let src = b"462:528,100:120";
        let expected = vec![462..528, 100..120];

        struct TestEncoder;
        impl Encoder for TestEncoder {
            type TargetOutput = Result<Vec<Element>>;
            type RecordOutput = Result<RecordData>;
            type EncodeOutput = Result<Vec<RecordData>>;
            fn encode_target(
                &self,
                _id: &[u8],
                _kmer_seq_len: Option<usize>,
            ) -> Self::TargetOutput {
                Ok(Vec::new())
            }
            fn encode_multiple(
                &mut self,
                _paths: &[PathBuf],
                _parallel: bool,
            ) -> Self::EncodeOutput {
                Ok(Vec::new())
            }
            fn encode<P: AsRef<Path>>(&mut self, _path: P) -> Self::EncodeOutput {
                Ok(Vec::new())
            }
            fn encode_record(&self, _id: &[u8], _seq: &[u8], _qual: &[u8]) -> Self::RecordOutput {
                Ok(RecordData::default())
            }
        }

        assert_eq!(TestEncoder::parse_target_from_id(src).unwrap(), expected);

        // Test case 2: Empty input
        let src = b"";
        let expected: Vec<Range<usize>> = Vec::new();
        assert_eq!(TestEncoder::parse_target_from_id(src).unwrap(), expected);

        // Test case 3: Invalid input (missing colon)
        let src = b"462528,100:120";
        assert!(TestEncoder::parse_target_from_id(src).is_err());

        // Test case 4: Invalid input (invalid number)
        let src = b"462:528,100:abc";
        assert!(TestEncoder::parse_target_from_id(src).is_err());
    }
}