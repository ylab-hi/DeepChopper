use anyhow::{anyhow, Result};
use derive_builder::Builder;
use std::ops::Range;
use std::path::Path;

use anyhow::Context;
use bio::utils::Interval; // 0-based, half-open interval [1, 10)
use ndarray::{concatenate, stack, Axis, Zip};

use ndarray::{Array2, Array3};

use needletail::{parse_fastx_file, Sequence};
use rayon::prelude::*;

use crate::error::EncodingError;
use crate::kmer::{generate_kmers_table, KmerTable};

// @462:528|738735b7-2105-460e-9e56-da980ef816c2+4f605fb4-4107-4827-9aed-9448d02834a8
// CGTTGGTGGTGTTCAGTTGTGGCGGTTGCTGGTCAGTAACAGCCAAGATGCTGCGGAATCTGCTGGCTTACCGTCAGATTGGGCAGAGGACGATAAGCACTGCTTCCCGCAGGCATTTTAAAAATAAAGTTCCGGAGAAGCAAAACTGTTCCAGGAGGATGATGAAATTCCACTGTATCTAAAAGGGTAGGGTAGCTGATGCCCTCCTGTATAGAGCCACCATGATCTTACAGTTGGTGGAACAGCATATGCCATATATGAGCTGGCTGTGGCTTCATTTCCCAAGAAGCAGGAGTGACTTTCAGCTTTATCTCCAGCAATTGCTTGGTCAGTTTTTCATTCAGCTCTCTATGGACCAGTAATCTGATAAATAACCGAGCTCTTCTTTGGGGATCAATATTTATTGATTGTAGTAACTGCCACCAATAAAGCAGTCTTTACCATGAAAAAAAAAAAAAAAAATCCCCCTACCCCTCTCTCCCAACTTATCCATACACAACCTGCCCCTCCAACCTCTTTCTAAACCCTTGGCGCCTCGGAGGCGTTCAGCTGCTTCAAGATGAAGCTGAACATCTTCCTTCCCAGCCACTGGCTGCCAGAAACTCATTGAAGTGGACGATGAACGCAAACTTCTGCACTTTCTATGAGAAGCGTATGGCCACAGAAGTTGCTGCTGACGCTTTGGGTGAAGAATGGAAGGGTTATGTGGTCCGAATCAGTGGTGGGAACGACAAACAAGGTTTCCCCATGAAGCAGGGTGTCTTGACCCATGGCCGTGTCCGCCTGCTACTGAGTAAGGGGCATTCCTGTTACAGACCAAGGAGAACTGGAGAAAGAAAGAGAAAATCAGTTCGTGGTTGCATTGTGGATGCAATCTGAGCGTTCTCAACTTGGTTATTGTAAAAAGGAGAGAAGGATATTCCTGGACTGACTGATACTACAGTGCCTCGCCGCCTGGGCCCCAAAAGAGCTAGCAGAATCCGCAAACTTTTCAATCTCTCTAAAAAGAAGATGATGTCCGCCAGTATCGTTGTAAGAAAGCCCTAAAATAAAGAAGGTAAGAAACCTAGGACCAAAGCACCCAAGATTCAGCGTCTGTTACTCCACGTGTCCTGCAGCACAAACGGCGGCGTATTGCTCTGAAGAAGCAGCGTACCAAGAAAAATAAAAGAAGAGGCTGCAGAATATGCTAAACTTTTGGCCTAGAGAATGAAGGAGGCTAAGGAGAAGCGCCAGGAACAAATTGCGAAGAGACGCAGACTTTCCTCTCTGCGGGACTCTACTTCTAAGTCTGAATCCAGTCAGAAATAAGATTTTTTGAGTAACAAATAATAAGATCGGGACTCTGA
// +
// 3A88;IMSJ872377DIJRSRRQRSSRSSSSSSSSGECCKLDIDDGLQRSRRRROSSPRRNOOSSEBCDEJSQKHHJSSSSSSSMMMPSSSSSRJ97677;<SSSSSRRSSSSSSSSSSSSJJKSSSSSSHFFBFGLBCBC<OPLMOP?KSIII6435@ESSSSKSSSSPSSSSD?22275@DB;(((478GIIJKMSIFEFKFA2-)&&''=ALPQQSSRSS,,;>SSSSSSSSSSSSOKGKOSSSSSSSQFLHGISSSSIGGHSSSSFFB.AGA0<AKLM9SSPLLMKMKLJJ..-02::<=0,+-)&&-:?BEFHNLIBA>>E89SSSSSASSQPSOPLMHG7788SSSCB==BCKLMPPPQQKIINRSSSSSSSSSSSSSSSSSPSRPOPPGGCH,,CEH1*%289<DACCRGGHISSSSSSSSQRQRSSSSSMQSRRRFFPPPPPPPPPPPPPPPPP--'.,,/42$))(('')'0314319HFF2104/)*+&#""33%%%(%%$##"""""""&..05%$((*(*36FSSSSSSS8794555HJI0///?SSSSSSSSSSSSSFD110AHKHKKJMJNOPS@;@@@HMQSSLMSOFC>546:<JNSIIIIKJJKSSSSSROO+(((--,1BLJKKSSSSSSSSSSSSSPKJLPSSSSSSSSSMFGPS22116559IIIISQQSSSSSSSSSSSSSQPMB651.13SSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSCC>.1ABR+(96;>SSRSQPPPSSSSSSRQMQNQRSSSSSSSSSSMSSSSSMKICDSRQPIF>>>?CL0GHLNMC=<;DBCCBBBCDABBSSSSSRRSSQRMNPPROHGBCFBBBAGGGL@OEC>53038=JSSSROJGKIIHGDDFLOOOSSIBCENLNENFFSSSSNMLF@JSSSPPRQR;:989FLPDBA00//7>PSSSSSSMIHHIRSSJGGIKKPPRRSED;:;?JO=::EKIGBD'&),QQSSSN>S/0KJJIMNORLH@6679FCF5556OOORNNLRQSSPSIII>4HF6A?=AHHSSPOKSLSN-,-?EFMSRSQ;;;DIIGEEIJOSLFE@?*)+-<CD?CFDA999AK;HIFGMQSSSSSSOKCBDSSSOLJ43115AJJGKKLOOMMNLOQSSSSI1,1CCOSS11:IIMSSSSSSSSSSSSOQRRSNJHSSSSSL/../<DEKMLLGPNOA?>>MOSSLKSSSBBCJRRRQSSSSS>76654;<BSONKIKK.--+0,,51+)+450045-,.5OSSHED777SSEEEJSSSSKKIGOOSSSSIIHJSSSSSSLIJOQSSJMSS;EFAA5**)+-2556BBJOM

pub type Element = i64;
pub type Matrix = Array2<Element>;
pub type Tensor = Array3<Element>;

pub const QUAL_OFFSET: u8 = 33;
pub const BASES: &[u8] = b"ATCGN";

#[derive(Debug)]
pub struct RecordData {
    id: Vec<u8>,
    seq: Vec<u8>,
    qual: Vec<u8>,
}

impl RecordData {
    pub fn new(id: Vec<u8>, seq: Vec<u8>, qual: Vec<u8>) -> Self {
        Self { id, seq, qual }
    }
}

impl From<(Vec<u8>, Vec<u8>, Vec<u8>)> for RecordData {
    fn from(data: (Vec<u8>, Vec<u8>, Vec<u8>)) -> Self {
        Self::new(data.0, data.1, data.2)
    }
}

#[derive(Debug, Builder, Default, Clone)]
pub struct FqEncoderOption {
    #[builder(default = "3")]
    kmer_size: u8,
    #[builder(default = "QUAL_OFFSET")]
    qual_offset: u8,
    #[builder(default = "BASES.to_vec()")]
    bases: Vec<u8>,
}

impl FqEncoderOption {
    pub fn new(kmer_size: u8, qual_offset: Option<u8>, base: Option<&[u8]>) -> Self {
        let base = base.unwrap_or(BASES);
        let qual_offset = qual_offset.unwrap_or(QUAL_OFFSET);
        Self {
            kmer_size,
            qual_offset,
            bases: base.to_vec(),
        }
    }
}

#[derive(Debug, Builder, Default)]
pub struct FqEncoder {
    pub option: FqEncoderOption,
    pub kmer_table: KmerTable,
}

impl FqEncoder {
    pub fn new(option: FqEncoderOption) -> Self {
        let kmer_table = generate_kmers_table(&option.bases, option.kmer_size);

        // rayon::ThreadPoolBuilder::new()
        //     .num_threads(1)
        //     .build_global()
        //     .unwrap();

        Self { option, kmer_table }
    }

    fn parse_target_from_id(src: &[u8]) -> Result<Vec<Range<usize>>> {
        // check empty input
        if src.is_empty() {
            return Ok(Vec::new());
        }

        // @462:528,100:120|738735b7-2105-460e-9e56-da980ef816c2+4f605fb4-4107-4827-9aed-9448d02834a8
        // removea content after |
        let number_part = src.split(|&c| c == b'|').next().unwrap();

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

    fn encode_target(&self, id: &[u8]) -> Result<Tensor> {
        let target = Self::parse_target_from_id(id).context("Failed to parse target from ID")?;
        let mut encoded_target = Tensor::zeros((1, target.len(), 2));

        // / Example of a parallel operation using Zip and par_apply from ndarray's parallel feature
        encoded_target
            .axis_iter_mut(Axis(0))
            .into_par_iter()
            .for_each(|mut subview| {
                Zip::from(subview.axis_iter_mut(Axis(0)))
                    .and(&target)
                    .for_each(|mut row, t| {
                        row[0] = t.start as Element;
                        row[1] = t.end as Element;
                    });
            });

        Ok(encoded_target)
    }

    fn encode_qual(qual: &[u8], socre_offset: Option<u8>) -> Vec<u8> {
        let offset = socre_offset.unwrap_or(33);
        // input is quality of fastq
        // 1. convert the quality to a score
        // 2. return the score
        qual.par_iter()
            .map(|&q| {
                // Convert ASCII to Phred score for Phred+33 encoding
                q - offset
            })
            .collect()
    }

    fn encoder_seq<'a>(&self, seq: &'a [u8]) -> Vec<&'a [u8]> {
        seq.kmers(self.option.kmer_size).collect()
    }

    pub fn encode_fq(
        &self,
        id: &[u8],
        seq: &[u8],
        qual: &[u8],
        max_width: usize,
    ) -> Result<(Tensor, Tensor)> {
        // 1.encode the sequence
        // 2.encode the quality

        // normalize to make sure all the bases are consistently capitalized and
        // that we remove the newlines since this is FASTA
        // change unknwon base to 'N'
        let _current_width = seq.len().saturating_sub(self.option.kmer_size as usize) + 1;

        // all assign -1 as missing value
        // let mut encoded_input = Tensor::from_elem((1, 2, max_width), -1);

        let encoded_seq = self.encoder_seq(seq);
        let mut encoded_seq_id = encoded_seq
            .par_iter()
            .map(|&s| {
                *self
                    .kmer_table
                    .get(s)
                    .context(format!("invalid kmer {}", String::from_utf8_lossy(s)))
                    .unwrap()
            })
            .collect::<Vec<_>>();
        encoded_seq_id.resize(max_width, -1);

        let matrix_seq_id = Matrix::from_shape_vec((1, max_width), encoded_seq_id)
            .context("invalid matrix shape herre ")?;

        let mut encoded_qual: Vec<_> = Self::encode_qual(qual, Some(33))
            .iter()
            .map(|x| *x as Element)
            .collect();
        encoded_qual.resize(max_width, -1);
        let matrix_qual = Matrix::from_shape_vec((1, max_width), encoded_qual)
            .context("invalid matrix shape here")?;

        let encoded_input = stack![Axis(1), matrix_seq_id, matrix_qual];
        let encoded_target = self.encode_target(id)?;

        Ok((encoded_input, encoded_target))
    }

    fn fetch_records<P: AsRef<Path>>(&self, path: P) -> Result<(Vec<RecordData>, usize)> {
        let mut reader = parse_fastx_file(path.as_ref()).context("valid path/file")?;
        let mut records = Vec::new();

        let mut max_seq_len = 0;

        while let Some(record) = reader.next() {
            let seqrec = record.context("invalid record")?;
            let id = seqrec.id();
            let seq = seqrec.normalize(false);
            let qual = seqrec.qual().context("invalid qual")?;

            let seq_len = seqrec.num_bases();
            let qual_len = qual.len();

            if seq_len < self.option.kmer_size as usize {
                continue;
            }

            assert_eq!(seq_len, qual_len);

            if seq_len > max_seq_len {
                max_seq_len = seq_len;
            }
            records.push((id.to_vec(), seq.to_vec(), qual.to_vec()).into());
        }

        Ok((records, max_seq_len))
    }

    pub fn encoder_fqs<P: AsRef<Path>>(&self, path: P) -> Result<(Tensor, Tensor)> {
        let (records, max_seq_len) = self.fetch_records(path)?;

        if max_seq_len < self.option.kmer_size as usize {
            return Err(EncodingError::SeqShorterThanKmer.into());
        }

        let max_width = max_seq_len - self.option.kmer_size as usize + 1;

        let data = records
            .par_iter()
            .map(|data| {
                let id = data.id.as_ref();
                let seq = data.seq.as_ref();
                let qual = data.qual.as_ref();

                self.encode_fq(id, seq, qual, max_width)
                    .context(format!(
                        "encode  fq read id {} error",
                        String::from_utf8_lossy(id)
                    ))
                    .unwrap()
            })
            .collect::<Vec<(Tensor, Tensor)>>();

        // Unzip the vector of tuples into two separate vectors
        let (inputs, targets): (Vec<Tensor>, Vec<Tensor>) = data.into_iter().unzip();

        // Here's the critical adjustment: Ensure inputs:a  list of (1, 2, shape) and targets a list of shaped (1, class, 2) and stack them
        let inputs_tensor = concatenate(
            Axis(0),
            &inputs.iter().map(|a| a.view()).collect::<Vec<_>>(),
        )
        .context("Failed to stack inputs")?;
        let targets_tensor = concatenate(
            Axis(0),
            &targets.iter().map(|a| a.view()).collect::<Vec<_>>(),
        )
        .context("Failed to stack targets")?;

        // concatenate the encoded input and target
        Ok((inputs_tensor, targets_tensor))
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

        assert_eq!(FqEncoder::parse_target_from_id(src).unwrap(), expected);

        // Test case 2: Empty input
        let src = b"";
        let expected: Vec<Range<usize>> = Vec::new();
        assert_eq!(FqEncoder::parse_target_from_id(src).unwrap(), expected);

        // Test case 3: Invalid input (missing colon)
        let src = b"462528,100:120";
        assert!(FqEncoder::parse_target_from_id(src).is_err());

        // Test case 4: Invalid input (invalid number)
        let src = b"462:528,100:abc";
        assert!(FqEncoder::parse_target_from_id(src).is_err());
    }

    #[test]
    fn test_encode_fqs() {
        let option = FqEncoderOptionBuilder::default()
            .kmer_size(3)
            .build()
            .unwrap();
        let encoder = FqEncoder::new(option);
        let (_input, target) = encoder.encoder_fqs("tests/data/test.fq.gz").unwrap();
        assert_eq!(target[[0, 0, 0]], 462);
        assert_eq!(target[[0, 0, 1]], 528);
    }
}
