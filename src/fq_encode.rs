use anyhow::{anyhow, Result};
use derive_builder::Builder;
use noodles::fastq;
use std::fmt::Display;
use std::fs::File;
use std::io::BufReader;
use std::ops::Range;
use std::path::{Path, PathBuf};

use anyhow::Context;
use ndarray::{concatenate, s, stack, Axis, Zip};

use needletail::Sequence;
use pyo3::prelude::*;
use rayon::prelude::*;

use crate::error::EncodingError;
use crate::kmer::{generate_kmers_table, to_kmer_target_region};
use crate::types::{Element, Id2KmerTable, Kmer2IdTable, Matrix, Tensor};

mod option;
mod record;
mod stat;

pub use option::*;
pub use record::*;
pub use stat::*;

use log::info;

// @462:528|738735b7-2105-460e-9e56-da980ef816c2+4f605fb4-4107-4827-9aed-9448d02834a8
// CGTTGGTGGTGTTCAGTTGTGGCGGTTGCTGGTCAGTAACAGCCAAGATGCTGCGGAATCTGCTGGCTTACCGTCAGATTGGGCAGAGGACGATAAGCACTGCTTCCCGCAGGCATTTTAAAAATAAAGTTCCGGAGAAGCAAAACTGTTCCAGGAGGATGATGAAATTCCACTGTATCTAAAAGGGTAGGGTAGCTGATGCCCTCCTGTATAGAGCCACCATGATCTTACAGTTGGTGGAACAGCATATGCCATATATGAGCTGGCTGTGGCTTCATTTCCCAAGAAGCAGGAGTGACTTTCAGCTTTATCTCCAGCAATTGCTTGGTCAGTTTTTCATTCAGCTCTCTATGGACCAGTAATCTGATAAATAACCGAGCTCTTCTTTGGGGATCAATATTTATTGATTGTAGTAACTGCCACCAATAAAGCAGTCTTTACCATGAAAAAAAAAAAAAAAAATCCCCCTACCCCTCTCTCCCAACTTATCCATACACAACCTGCCCCTCCAACCTCTTTCTAAACCCTTGGCGCCTCGGAGGCGTTCAGCTGCTTCAAGATGAAGCTGAACATCTTCCTTCCCAGCCACTGGCTGCCAGAAACTCATTGAAGTGGACGATGAACGCAAACTTCTGCACTTTCTATGAGAAGCGTATGGCCACAGAAGTTGCTGCTGACGCTTTGGGTGAAGAATGGAAGGGTTATGTGGTCCGAATCAGTGGTGGGAACGACAAACAAGGTTTCCCCATGAAGCAGGGTGTCTTGACCCATGGCCGTGTCCGCCTGCTACTGAGTAAGGGGCATTCCTGTTACAGACCAAGGAGAACTGGAGAAAGAAAGAGAAAATCAGTTCGTGGTTGCATTGTGGATGCAATCTGAGCGTTCTCAACTTGGTTATTGTAAAAAGGAGAGAAGGATATTCCTGGACTGACTGATACTACAGTGCCTCGCCGCCTGGGCCCCAAAAGAGCTAGCAGAATCCGCAAACTTTTCAATCTCTCTAAAAAGAAGATGATGTCCGCCAGTATCGTTGTAAGAAAGCCCTAAAATAAAGAAGGTAAGAAACCTAGGACCAAAGCACCCAAGATTCAGCGTCTGTTACTCCACGTGTCCTGCAGCACAAACGGCGGCGTATTGCTCTGAAGAAGCAGCGTACCAAGAAAAATAAAAGAAGAGGCTGCAGAATATGCTAAACTTTTGGCCTAGAGAATGAAGGAGGCTAAGGAGAAGCGCCAGGAACAAATTGCGAAGAGACGCAGACTTTCCTCTCTGCGGGACTCTACTTCTAAGTCTGAATCCAGTCAGAAATAAGATTTTTTGAGTAACAAATAATAAGATCGGGACTCTGA
// +
// 3A88;IMSJ872377DIJRSRRQRSSRSSSSSSSSGECCKLDIDDGLQRSRRRROSSPRRNOOSSEBCDEJSQKHHJSSSSSSSMMMPSSSSSRJ97677;<SSSSSRRSSSSSSSSSSSSJJKSSSSSSHFFBFGLBCBC<OPLMOP?KSIII6435@ESSSSKSSSSPSSSSD?22275@DB;(((478GIIJKMSIFEFKFA2-)&&''=ALPQQSSRSS,,;>SSSSSSSSSSSSOKGKOSSSSSSSQFLHGISSSSIGGHSSSSFFB.AGA0<AKLM9SSPLLMKMKLJJ..-02::<=0,+-)&&-:?BEFHNLIBA>>E89SSSSSASSQPSOPLMHG7788SSSCB==BCKLMPPPQQKIINRSSSSSSSSSSSSSSSSSPSRPOPPGGCH,,CEH1*%289<DACCRGGHISSSSSSSSQRQRSSSSSMQSRRRFFPPPPPPPPPPPPPPPPP--'.,,/42$))(('')'0314319HFF2104/)*+&#""33%%%(%%$##"""""""&..05%$((*(*36FSSSSSSS8794555HJI0///?SSSSSSSSSSSSSFD110AHKHKKJMJNOPS@;@@@HMQSSLMSOFC>546:<JNSIIIIKJJKSSSSSROO+(((--,1BLJKKSSSSSSSSSSSSSPKJLPSSSSSSSSSMFGPS22116559IIIISQQSSSSSSSSSSSSSQPMB651.13SSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSCC>.1ABR+(96;>SSRSQPPPSSSSSSRQMQNQRSSSSSSSSSSMSSSSSMKICDSRQPIF>>>?CL0GHLNMC=<;DBCCBBBCDABBSSSSSRRSSQRMNPPROHGBCFBBBAGGGL@OEC>53038=JSSSROJGKIIHGDDFLOOOSSIBCENLNENFFSSSSNMLF@JSSSPPRQR;:989FLPDBA00//7>PSSSSSSMIHHIRSSJGGIKKPPRRSED;:;?JO=::EKIGBD'&),QQSSSN>S/0KJJIMNORLH@6679FCF5556OOORNNLRQSSPSIII>4HF6A?=AHHSSPOKSLSN-,-?EFMSRSQ;;;DIIGEEIJOSLFE@?*)+-<CD?CFDA999AK;HIFGMQSSSSSSOKCBDSSSOLJ43115AJJGKKLOOMMNLOQSSSSI1,1CCOSS11:IIMSSSSSSSSSSSSOQRRSNJHSSSSSL/../<DEKMLLGPNOA?>>MOSSLKSSSBBCJRRRQSSSSS>76654;<BSONKIKK.--+0,,51+)+450045-,.5OSSHED777SSEEEJSSSSKKIGOOSSSSIIHJSSSSSSLIJOQSSJMSS;EFAA5**)+-2556BBJOM

#[pyclass]
#[derive(Debug, Builder, Default, Clone)]
pub struct FqEncoder {
    pub option: FqEncoderOption,
    pub kmer2id_table: Kmer2IdTable,
    pub id2kmer_table: Id2KmerTable,
}

impl Display for FqEncoder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "FqEncoder {{ option: {} }}", self.option)
    }
}

impl FqEncoder {
    pub fn new(option: FqEncoderOption) -> Self {
        let kmer2id_table = generate_kmers_table(&option.bases, option.kmer_size);
        let id2kmer_table: Id2KmerTable = kmer2id_table
            .par_iter()
            .map(|(kmer, id)| (*id, kmer.clone()))
            .collect();

        // rayon::ThreadPoolBuilder::new()
        //     .num_threads(1)
        //     .build_global()
        //     .unwrap();

        Self {
            option,
            kmer2id_table,
            id2kmer_table,
        }
    }

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

    fn encode_target(&self, id: &[u8]) -> Result<Tensor> {
        let target = Self::parse_target_from_id(id).context("Failed to parse target from ID")?;
        let kmer_target = target
            .par_iter()
            .map(|range| to_kmer_target_region(range, self.option.kmer_size as usize, None))
            .collect::<Result<Vec<Range<usize>>>>()?;

        if self.option.vectorized_target {
            let mut encoded_target = Tensor::zeros((1, target.len(), self.option.max_width));

            // Example of a parallel operation using Zip and par_apply from ndarray's parallel feature
            encoded_target
                .axis_iter_mut(Axis(0))
                .into_par_iter()
                .for_each(|mut subview| {
                    Zip::from(subview.axis_iter_mut(Axis(0)))
                        .and(&kmer_target)
                        .for_each(|mut row, t| {
                            // Safe fill based on kmer_target, assuming it's within bounds
                            if t.start < t.end && t.end <= row.len() {
                                row.slice_mut(s![t.start..t.end]).fill(1);
                            }
                        });
                });
            return Ok(encoded_target);
        }

        let mut encoded_target = Tensor::zeros((1, target.len(), 2));

        encoded_target
            .axis_iter_mut(Axis(0))
            .into_par_iter()
            .for_each(|mut subview| {
                Zip::from(subview.axis_iter_mut(Axis(0)))
                    .and(&kmer_target)
                    .for_each(|mut row, t| {
                        row[0] = t.start as Element;
                        row[1] = t.end as Element;
                    });
            });

        Ok(encoded_target)
    }

    fn encode_qual(&self, qual: &[u8]) -> (Vec<Element>, Vec<Element>) {
        // input is quality of fastq
        // 1. convert the quality to a score
        // 2. return the score
        let encoded_qual: Vec<u8> = qual
            .par_iter()
            .map(|&q| {
                // Convert ASCII to Phred score for Phred+33 encoding
                q - self.option.qual_offset
            })
            .collect();

        let encoded_kmer_qual: Vec<Element> = encoded_qual
            .kmers(self.option.kmer_size)
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

    fn encoder_seq<'a>(&self, seq: &'a [u8]) -> Vec<&'a [u8]> {
        seq.kmers(self.option.kmer_size).collect()
    }
    fn unpack_data_parallel(
        data: Vec<((Tensor, Tensor), Matrix)>,
    ) -> (Vec<Tensor>, Vec<Tensor>, Vec<Matrix>) {
        let (packed_tensors, elements): (Vec<_>, Vec<_>) = data.into_par_iter().unzip();
        let (tensors1, tensors2): (Vec<_>, Vec<_>) = packed_tensors.into_par_iter().unzip();
        (tensors1, tensors2, elements)
    }

    pub fn encode_fq(
        &self,
        id: &[u8],
        seq: &[u8],
        qual: &[u8],
    ) -> Result<((Tensor, Tensor), Matrix)> {
        println!("encoding record: {}", String::from_utf8_lossy(id));
        // 1.encode the sequence
        // 2.encode the quality

        // normalize to make sure all the bases are consistently capitalized and
        // that we remove the newlines since this is FASTA
        // change unknwon base to 'N'
        let current_width = seq.len().saturating_sub(self.option.kmer_size as usize) + 1;

        if current_width > self.option.max_width {
            return Err(anyhow!(
                "invalid current_width: {} > max_width: {}",
                current_width,
                self.option.max_width
            ));
        }

        // encode the sequence
        let encoded_seq = self.encoder_seq(seq);

        let mut encoded_seq_id = encoded_seq
            .into_par_iter()
            .map(|s| {
                self.kmer2id_table
                    .get(s)
                    .ok_or(anyhow!("invalid kmer"))
                    .copied()
            })
            .collect::<Result<Vec<Element>>>()?;

        if encoded_seq_id.len() != current_width {
            return Err(anyhow!(
                "invalid encoded_seq_id length: {} != current_width: {}",
                encoded_seq_id.len(),
                current_width
            ));
        }

        encoded_seq_id.resize(self.option.max_width, -1);

        let matrix_seq_id = Matrix::from_shape_vec((1, self.option.max_width), encoded_seq_id)
            .context("invalid matrix shape herre ")?;

        // encode the quality
        let (mut encoded_qual, mut encoded_kmer_qual) = self.encode_qual(qual);
        encoded_kmer_qual.resize(self.option.max_width, -1);

        encoded_qual.resize(self.option.max_seq_len, -1);
        let matrix_qual = Matrix::from_shape_vec((1, self.option.max_seq_len), encoded_qual)
            .context("invalid matrix shape here")?;

        let matrix_kmer_qual =
            Matrix::from_shape_vec((1, self.option.max_width), encoded_kmer_qual)
                .context("invalid matrix shape here")?;
        // assemble the input and target
        let input_tensor = stack![Axis(1), matrix_seq_id, matrix_kmer_qual];
        let target_tensor = self.encode_target(id)?;

        Ok(((input_tensor, target_tensor), matrix_qual))
    }

    fn fetch_records<P: AsRef<Path>>(&mut self, path: P) -> Result<Vec<RecordData>> {
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

            if seq_len < self.option.kmer_size as usize {
                continue;
            }

            if seq_len != qual_len {
                return Err(anyhow!(
                    "record: id {} seq_len != qual_len",
                    String::from_utf8_lossy(id)
                ));
            }

            if seq_len > self.option.max_seq_len {
                self.option.max_seq_len = seq_len;
            }
            records.push((id.to_vec(), seq.to_vec(), qual.to_vec()).into());
        }

        if self.option.max_seq_len < self.option.kmer_size as usize {
            return Err(EncodingError::SeqShorterThanKmer.into());
        }

        let max_width = self.option.max_seq_len - self.option.kmer_size as usize + 1;

        if max_width > self.option.max_width {
            self.option.max_width = max_width;
        }

        info!("total records: {}", records.len());
        info!("max_seq_len: {}", self.option.max_seq_len);
        info!("max_width: {}", self.option.max_width);
        Ok(records)
    }

    pub fn encode_fq_path<P: AsRef<Path>>(
        &mut self,
        path: P,
    ) -> Result<((Tensor, Tensor), Matrix)> {
        let records = self.fetch_records(path)?;

        let data: Vec<((Tensor, Tensor), Matrix)> = records
            .into_par_iter()
            .filter_map(|data| {
                let id = data.id.as_ref();
                let seq = data.seq.as_ref();
                let qual = data.qual.as_ref();

                match self.encode_fq(id, seq, qual).context(format!(
                    "encode fq read id {} error",
                    String::from_utf8_lossy(id)
                )) {
                    Ok(result) => Some(result),
                    Err(_e) => None,
                }
            })
            .collect();

        info!("encoded records: {}", data.len());

        // Unzip the vector of tuples into two separate vectors
        let (inputs, targets, quals): (Vec<Tensor>, Vec<Tensor>, Vec<Matrix>) =
            Self::unpack_data_parallel(data);

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

        let quals_matrix =
            concatenate(Axis(0), &quals.iter().map(|a| a.view()).collect::<Vec<_>>())
                .context("Failed to stack quals")?;

        // concatenate the encoded input and target
        Ok(((inputs_tensor, targets_tensor), quals_matrix))
    }

    pub fn encode_fq_paths(&self, paths: &[PathBuf]) -> Result<((Tensor, Tensor), Matrix)> {
        let result = paths
            .into_par_iter()
            .map(|path| {
                let mut encoder = self.clone();
                encoder.encode_fq_path(path)
            })
            .collect::<Result<Vec<_>>>()?;

        let (inputs, targets, quals): (Vec<Tensor>, Vec<Tensor>, Vec<Matrix>) =
            Self::unpack_data_parallel(result);

        let inputs_tensor = concatenate(
            Axis(0),
            &inputs.iter().map(|a| a.view()).collect::<Vec<_>>(),
        )
        .context("failed to stack inputs")?;

        let targets_tensor = concatenate(
            Axis(0),
            &targets.iter().map(|a| a.view()).collect::<Vec<_>>(),
        )
        .context("failed to stack targets")?;

        let quals_matrix =
            concatenate(Axis(0), &quals.iter().map(|a| a.view()).collect::<Vec<_>>())
                .context("Failed to stack quals")?;
        Ok(((inputs_tensor, targets_tensor), quals_matrix))
    }
}

#[cfg(test)]
mod tests {
    use ndarray::Array1;

    use crate::kmer::{kmerids_to_seq, to_original_targtet_region};

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
        let mut encoder = FqEncoder::new(option);
        let ((_input, target), _qual) = encoder.encode_fq_path("tests/data/one_record.fq").unwrap();
        let k = 3;

        let actual = 462..528;

        let kmer_target = to_kmer_target_region(&actual, k, None).unwrap();
        let expected_target = to_original_targtet_region(&kmer_target, k);

        assert_eq!(expected_target, actual);

        assert_eq!(target[[0, 0, 0]], kmer_target.start as Element);
        assert_eq!(target[[0, 0, 1]], kmer_target.end as Element);
    }

    #[test]
    fn test_encode_fqs_vectorized_target() {
        let option = FqEncoderOptionBuilder::default()
            .kmer_size(3)
            .vectorized_target(true)
            .build()
            .unwrap();

        let mut encoder = FqEncoder::new(option);
        let ((_input, target), _qual) = encoder.encode_fq_path("tests/data/one_record.fq").unwrap();

        let k = 3;
        let actual = 462..528;
        // let actual_target_seq = _input.slice(s![0, 0, actual.clone()]);

        let kmer_target = to_kmer_target_region(&actual, k, None).unwrap();
        let expected_target = to_original_targtet_region(&kmer_target, k);

        assert_eq!(expected_target, actual);

        let expected_vectorized_target = Array1::<Element>::from_elem(kmer_target.len(), 1);

        assert_eq!(
            target.slice(s![0, 0, kmer_target.clone()]),
            expected_vectorized_target
        );

        // construct sequence from list of kmerid
        let actual_target_seq =
            b"TCCCCCTACCCCTCTCTCCCAACTTATCCATACACAACCTGCCCCTCCAACCTCTTTCTAAACCCT";
        let kmerids = _input.slice(s![0, 0, kmer_target]).to_vec();
        let kmer_seq: Vec<u8> = kmerids_to_seq(&kmerids, encoder.id2kmer_table).unwrap();
        assert_eq!(actual_target_seq, kmer_seq.as_slice());
    }

    #[test]
    fn test_encode_fqs_vectorized_target_with_small_max_width() {
        let option = FqEncoderOptionBuilder::default()
            .kmer_size(3)
            .vectorized_target(true)
            .max_width(100)
            .build()
            .unwrap();

        let mut encoder = FqEncoder::new(option);
        let ((input, target), qual) = encoder.encode_fq_path("tests/data/one_record.fq").unwrap();

        assert_eq!(input.shape(), &[1, 2, 1347]);
        assert_eq!(target.shape(), &[1, 1, 1347]);
        assert_eq!(qual.shape(), &[1, 1349]);
    }

    #[test]
    fn test_encode_fqs_vectorized_target_with_large_max_width() {
        let option = FqEncoderOptionBuilder::default()
            .kmer_size(3)
            .vectorized_target(true)
            .max_width(2000)
            .max_seq_len(2000)
            .build()
            .unwrap();

        let mut encoder = FqEncoder::new(option);
        let ((input, target), qual) = encoder.encode_fq_path("tests/data/one_record.fq").unwrap();

        assert_eq!(input.shape(), &[1, 2, 2000]);
        assert_eq!(target.shape(), &[1, 1, 2000]);
        assert_eq!(qual.shape(), &[1, 2000]);
    }
    #[test]
    fn test_encode_fqs_vectorized_target_with_large_max_width_for_large_size_fq() {
        let option = FqEncoderOptionBuilder::default()
            .kmer_size(3)
            .vectorized_target(true)
            .build()
            .unwrap();

        let mut encoder = FqEncoder::new(option);
        let ((input, target), qual) = encoder
            .encode_fq_path("tests/data/twenty_five_records.fq")
            .unwrap();

        assert_eq!(input.shape(), &[25, 2, 4741]);
        assert_eq!(target.shape(), &[25, 1, 4741]);
        assert_eq!(qual.shape(), &[25, 4743]);
    }

    #[test]
    fn test_encode_fq_paths() {
        let option = FqEncoderOptionBuilder::default()
            .kmer_size(3)
            .max_width(15000)
            .max_seq_len(15000)
            .vectorized_target(true)
            .build()
            .unwrap();

        let encoder = FqEncoder::new(option);
        let paths = vec![
            "tests/data/twenty_five_records.fq",
            "tests/data/1000_records.fq",
        ]
        .into_iter()
        .map(PathBuf::from)
        .collect::<Vec<_>>();

        let ((inputs, targets), quals) = encoder.encode_fq_paths(&paths).unwrap();

        assert_eq!(inputs.shape(), &[1025, 2, 15000]);
        assert_eq!(targets.shape(), &[1025, 1, 15000]);
        assert_eq!(quals.shape(), &[1025, 15000]);
    }
}
