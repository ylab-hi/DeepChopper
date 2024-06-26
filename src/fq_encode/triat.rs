use anyhow::Result;
use anyhow::{anyhow, Context};
use log::info;
use needletail::Sequence;
use rayon::prelude::*;
use std::ops::Range;
use std::path::{Path, PathBuf};

use crate::output;
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

        // TODO: add code to parse negative case  and then return [0, 0) <03-02-24, Yangyang Li>
        // if no | in the id, return [0, 0)
        if !src.contains(&b'|') {
            return Ok(vec![0..0]);
        }
        // @738735b7-2105-460e-9e56-da980ef816c2+4f605fb4-4107-4827-9aed-9448d02834a8|462:528,100:120
        // remove content before |
        let number_part = src
            .split(|&c| c == b'|')
            .last()
            .context("Failed to get number part")?;

        let result = number_part
            .par_split(|&c| c == b',')
            .map(|target| {
                let mut parts = target.split(|&c| c == b':');
                let start: usize =
                    lexical::parse(parts.next().ok_or(anyhow!("parse number error"))?)?;
                let end: usize =
                    lexical::parse(parts.next().ok_or(anyhow!("parse number error"))?)?;
                Ok(start..end)
            })
            .collect::<Result<Vec<_>>>();

        if result.is_err() {
            Ok(vec![0..0])
        } else {
            result
        }
    }

    fn fetch_records<P: AsRef<Path>>(&mut self, path: P, kmer_size: u8) -> Result<Vec<RecordData>> {
        info!("fetching records from {}", path.as_ref().display());
        let _records = output::read_noodel_records_from_fq_or_zip_fq(path)?;

        let records: Vec<RecordData> = _records
            .into_par_iter()
            .filter_map(|record| {
                let id = record.definition().name();
                let seq = record.sequence();
                let normalized_seq = seq.normalize(false);
                let qual = record.quality_scores();
                let seq_len = normalized_seq.len();
                let qual_len = qual.len();

                if kmer_size > 0 && seq_len < kmer_size as usize {
                    return None::<RecordData>;
                }

                if seq_len != qual_len {
                    // NOTE: it seems like log mes does not work well with rayon paralllel iterator  <02-26-24, Yangyang Li>
                    // warn!(
                    //     "record: id {} seq_len != qual_len",
                    //     String::from_utf8_lossy(id)
                    // );
                    return None;
                }

                Some((id.to_vec(), seq.to_vec(), qual.to_vec()).into())
            })
            .collect();

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

    fn encoder_seq<'a>(
        &self,
        seq: &'a [u8],
        kmer_size: u8,
        overlap: bool,
    ) -> Result<Vec<&'a [u8]>> {
        if overlap {
            return Ok(seq.par_windows(kmer_size as usize).collect());
        }

        if kmer_size as usize > seq.len() {
            return Err(anyhow!("kmer_size is larger than seq length"));
        }

        Ok(seq.par_chunks(kmer_size as usize).collect())
    }
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn test_parse_target_from_id() {
        // Test case 1: Valid input
        let src = b"@test_name|462:528,100:120";
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

        let src = b"738735b7-2105-460e-9e56-da980ef816c2+4f605fb4-4107-4827-9aed-9448d02834a8";
        let result = TestEncoder::parse_target_from_id(src).unwrap();
        assert_eq!(result, vec![0..0]);
    }
}
