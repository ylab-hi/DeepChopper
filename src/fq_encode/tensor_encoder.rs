use std::{
    fmt::Display,
    fs::File,
    io::BufReader,
    ops::Range,
    path::{Path, PathBuf},
};

use anyhow::{anyhow, Context, Result};
use derive_builder::Builder;
use log::info;
use ndarray::{concatenate, s, stack, Axis, Zip};
use noodles::fastq;
use pyo3::prelude::*;

use crate::{
    error::EncodingError,
    fq_encode::RecordData,
    kmer::{generate_kmers_table, to_kmer_target_region},
    types::{Element, Id2KmerTable, Kmer2IdTable, Matrix, Tensor},
};

use super::{triat::Encoder, FqEncoderOption};
use needletail::Sequence;
use rayon::prelude::*;

#[pyclass]
#[derive(Debug, Builder, Default, Clone)]
#[builder(build_fn(skip))] // Specify custom build function
pub struct TensorEncoder {
    pub option: FqEncoderOption,

    #[pyo3(get, set)]
    #[builder(default = "0")]
    pub tensor_max_width: usize, // control width of input and target tensor

    #[pyo3(get, set)]
    #[builder(default = "0")]
    pub tensor_max_seq_len: usize, // control width of original qual matrix

    #[pyo3(get, set)]
    pub kmer2id_table: Kmer2IdTable,

    #[pyo3(get, set)]
    pub id2kmer_table: Id2KmerTable,
}

impl Display for TensorEncoder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "FqEncoder {{ option: {} }}", self.option)
    }
}

fn unpack_data_parallel(
    data: Vec<((Tensor, Tensor), Matrix)>,
) -> (Vec<Tensor>, Vec<Tensor>, Vec<Matrix>) {
    let (packed_tensors, elements): (Vec<_>, Vec<_>) = data.into_par_iter().unzip();
    let (tensors1, tensors2): (Vec<_>, Vec<_>) = packed_tensors.into_par_iter().unzip();
    (tensors1, tensors2, elements)
}

impl TensorEncoderBuilder {
    pub fn build(&self) -> Result<TensorEncoder> {
        let option = self.option.clone().unwrap_or_default();
        Ok(TensorEncoder::new(
            option,
            self.tensor_max_width,
            self.tensor_max_seq_len,
        ))
    }
}

impl TensorEncoder {
    pub fn build_custom(&self) -> Result<TensorEncoder> {
        println!("building custom TensorEncoder");
        // Extract values from builder, handling defaults and unwrapping Options
        let option = self.option.clone();
        let tensor_max_width = self.tensor_max_width;
        let tensor_max_seq_len = self.tensor_max_seq_len;

        // Your existing logic to initialize kmer2id_table and id2kmer_table
        let kmer2id_table = generate_kmers_table(&option.bases, option.kmer_size);
        let id2kmer_table: Id2KmerTable = kmer2id_table
            .par_iter()
            .map(|(kmer, id)| (*id, kmer.clone()))
            .collect();

        // Construct and return the TensorEncoder instance
        Ok(TensorEncoder {
            option,
            tensor_max_width,
            tensor_max_seq_len,
            kmer2id_table,
            id2kmer_table,
        })
    }

    pub fn new(
        option: FqEncoderOption,
        max_width: Option<usize>,
        max_seq_len: Option<usize>,
    ) -> Self {
        let kmer2id_table = generate_kmers_table(&option.bases, option.kmer_size);
        let id2kmer_table: Id2KmerTable = kmer2id_table
            .par_iter()
            .map(|(kmer, id)| (*id, kmer.clone()))
            .collect();

        let tensor_max_width = max_width.unwrap_or_default();
        let tensor_max_seq_len = max_seq_len.unwrap_or_default();

        Self {
            option,
            kmer2id_table,
            id2kmer_table,
            tensor_max_width,
            tensor_max_seq_len,
        }
    }
}

impl Encoder for TensorEncoder {
    type TargetOutput = Result<Tensor>;
    type RecordOutput = Result<((Tensor, Tensor), Matrix)>;
    type EncodeOutput = Result<((Tensor, Tensor), Matrix)>;

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

            if seq_len > self.tensor_max_seq_len {
                self.tensor_max_seq_len = seq_len;
            }
            records.push((id.to_vec(), seq.to_vec(), qual.to_vec()).into());
        }

        if self.tensor_max_seq_len < self.option.kmer_size as usize {
            return Err(EncodingError::SeqShorterThanKmer.into());
        }

        let max_width = self.tensor_max_seq_len - self.option.kmer_size as usize + 1;

        if max_width > self.tensor_max_width {
            self.tensor_max_width = max_width;
        }

        info!("total records: {}", records.len());
        info!("max_seq_len: {}", self.tensor_max_seq_len);
        info!("max_width: {}", self.tensor_max_width);
        Ok(records)
    }

    fn encode_target(&self, id: &[u8], _kmer_seq_len: Option<usize>) -> Self::TargetOutput {
        let target = Self::parse_target_from_id(id).context("Failed to parse target from ID")?;
        let kmer_target = target
            .par_iter()
            .map(|range| to_kmer_target_region(range, self.option.kmer_size as usize, None))
            .collect::<Result<Vec<Range<usize>>>>()?;

        if self.option.vectorized_target {
            let mut encoded_target = Tensor::zeros((1, target.len(), self.tensor_max_width));

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

    fn encode_record(&self, id: &[u8], seq: &[u8], qual: &[u8]) -> Self::RecordOutput {
        // println!("encoding record: {}", String::from_utf8_lossy(id));
        // 1.encode the sequence
        // 2.encode the quality

        // normalize to make sure all the bases are consistently capitalized and
        // that we remove the newlines since this is FASTA
        // change unknwon base to 'N'
        let current_width = seq.len().saturating_sub(self.option.kmer_size as usize) + 1;

        if current_width > self.tensor_max_width {
            return Err(anyhow!(
                "invalid current_width: {} > max_width: {}",
                current_width,
                self.tensor_max_width
            ));
        }

        // encode the sequence
        let encoded_seq = self.encoder_seq(seq, self.option.kmer_size, true)?;

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
                "invalid encoded_seq_id length: {} != current_width: {}, please make sure set max_width and max_seq_len",
                encoded_seq_id.len(),
                current_width
            ));
        }

        encoded_seq_id.resize(self.tensor_max_width, -1);

        let matrix_seq_id = Matrix::from_shape_vec((1, self.tensor_max_width), encoded_seq_id)
            .context("invalid matrix shape herre ")?;

        // encode the quality
        let (mut encoded_qual, mut encoded_kmer_qual) =
            self.encode_qual(qual, self.option.kmer_size, self.option.qual_offset);
        encoded_kmer_qual.resize(self.tensor_max_width, -1);

        encoded_qual.resize(self.tensor_max_seq_len, -1);
        let matrix_qual = Matrix::from_shape_vec((1, self.tensor_max_seq_len), encoded_qual)
            .context("invalid matrix shape here")?;

        let matrix_kmer_qual =
            Matrix::from_shape_vec((1, self.tensor_max_width), encoded_kmer_qual)
                .context("invalid matrix shape here")?;
        // assemble the input and target
        let input_tensor = stack![Axis(1), matrix_seq_id, matrix_kmer_qual];
        let target_tensor = self.encode_target(id, None)?;

        Ok(((input_tensor, target_tensor), matrix_qual))
    }

    fn encode<P: AsRef<Path>>(&mut self, path: P) -> Self::EncodeOutput {
        let records = self.fetch_records(path, self.option.kmer_size)?;

        let data: Vec<((Tensor, Tensor), Matrix)> = records
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

        // Unzip the vector of tuples into two separate vectors
        let (inputs, targets, quals): (Vec<Tensor>, Vec<Tensor>, Vec<Matrix>) =
            unpack_data_parallel(data);

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

        let (inputs, targets, quals): (Vec<Tensor>, Vec<Tensor>, Vec<Matrix>) =
            unpack_data_parallel(result);

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

    use crate::{
        fq_encode::FqEncoderOptionBuilder,
        kmer::{kmerids_to_seq, to_original_targtet_region},
    };

    use super::*;

    #[test]
    fn test_encode_fqs() {
        let option = FqEncoderOptionBuilder::default()
            .kmer_size(3)
            .build()
            .unwrap();
        let mut encoder = TensorEncoderBuilder::default()
            .option(option)
            .build()
            .unwrap();

        let ((_input, target), _qual) = encoder.encode("tests/data/one_record.fq").unwrap();
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

        let mut encoder = TensorEncoderBuilder::default()
            .option(option)
            .build()
            .unwrap();

        let ((_input, target), _qual) = encoder.encode("tests/data/one_record.fq").unwrap();

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
            .build()
            .unwrap();

        let mut encoder = TensorEncoderBuilder::default()
            .option(option)
            .tensor_max_width(100)
            .build()
            .unwrap();

        let ((input, target), qual) = encoder.encode("tests/data/one_record.fq").unwrap();

        assert_eq!(input.shape(), &[1, 2, 1347]);
        assert_eq!(target.shape(), &[1, 1, 1347]);
        assert_eq!(qual.shape(), &[1, 1349]);
    }

    #[test]
    fn test_encode_fqs_vectorized_target_with_large_max_width() {
        let option = FqEncoderOptionBuilder::default()
            .kmer_size(3)
            .vectorized_target(true)
            .build()
            .unwrap();

        let mut encoder = TensorEncoderBuilder::default()
            .option(option)
            .tensor_max_width(2000)
            .tensor_max_seq_len(2000)
            .build()
            .unwrap();

        let ((input, target), qual) = encoder.encode("tests/data/one_record.fq").unwrap();

        assert_eq!(input.shape(), &[1, 2, 2000]);
        assert_eq!(target.shape(), &[1, 1, 2000]);
        assert_eq!(qual.shape(), &[1, 2000]);
    }

    #[test]
    fn test_encode_fqs_in_a_row_vectorized_target_with_large_max_width() {
        let option = FqEncoderOptionBuilder::default()
            .kmer_size(3)
            .vectorized_target(true)
            .build()
            .unwrap();

        let mut tensor_encoder = TensorEncoderBuilder::default()
            .option(option)
            .tensor_max_width(2000)
            .tensor_max_seq_len(2000)
            .build()
            .unwrap();

        println!("{:?}", tensor_encoder);

        let ((input, target), qual) = tensor_encoder.encode("tests/data/one_record.fq").unwrap();
        assert_eq!(input.shape(), &[1, 2, 2000]);
        assert_eq!(target.shape(), &[1, 1, 2000]);
        assert_eq!(qual.shape(), &[1, 2000]);
    }

    #[test]
    fn test_encode_fq_paths() {
        let option = FqEncoderOptionBuilder::default()
            .kmer_size(3)
            .vectorized_target(true)
            .build()
            .unwrap();

        let mut encoder = TensorEncoderBuilder::default()
            .option(option)
            .tensor_max_width(15000)
            .tensor_max_seq_len(15000)
            .build()
            .unwrap();

        let paths = vec!["tests/data/twenty_five_records.fq"]
            .into_iter()
            .map(PathBuf::from)
            .collect::<Vec<_>>();

        let ((inputs, targets), quals) = encoder.encode_multiple(&paths, true).unwrap();

        assert_eq!(inputs.shape(), &[25, 2, 15000]);
        assert_eq!(targets.shape(), &[25, 1, 15000]);
        assert_eq!(quals.shape(), &[25, 15000]);
    }

    #[test]
    fn test_encode_fqs_vectorized_target_with_large_max_width_for_large_size_fq() {
        let option = FqEncoderOptionBuilder::default()
            .kmer_size(3)
            .vectorized_target(true)
            .build()
            .unwrap();

        let mut encoder = TensorEncoderBuilder::default()
            .option(option)
            .build()
            .unwrap();

        let ((input, target), qual) = encoder.encode("tests/data/twenty_five_records.fq").unwrap();

        assert_eq!(input.shape(), &[25, 2, 4741]);
        assert_eq!(target.shape(), &[25, 1, 4741]);
        assert_eq!(qual.shape(), &[25, 4743]);
    }
}
