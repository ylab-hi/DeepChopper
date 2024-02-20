use std::{
    fmt::Display,
    ops::Range,
    path::{Path, PathBuf},
    sync::Arc,
};

use arrow::array::{Array, Int32Builder, ListBuilder, RecordBatch, StringBuilder};
use arrow::datatypes::{DataType, Field, Schema};

use derive_builder::Builder;

use crate::{kmer::to_kmer_target_region, types::Element};

use super::{triat::Encoder, FqEncoderOption};
use anyhow::{anyhow, Context, Result};
use pyo3::prelude::*;
use rayon::prelude::*;

#[derive(Debug, Builder, Default)]
pub struct ParquetData {
    pub id: String,                // id
    pub kmer_seq: Vec<String>,     // kmer_seq
    pub kmer_qual: Vec<Element>,   // kmer_qual
    pub kmer_target: Vec<Element>, // kmer_target
    pub qual: Vec<Element>,        // qual
}

#[pyclass]
#[derive(Debug, Builder, Default, Clone)]
pub struct ParquetEncoder {
    pub option: FqEncoderOption,
}

impl ParquetEncoder {
    pub fn new(option: FqEncoderOption) -> Self {
        Self { option }
    }
}

impl Display for ParquetEncoder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "FqEncoder {{ option: {} }}", self.option)
    }
}

impl Encoder for ParquetEncoder {
    type TargetOutput = Result<Vec<Element>>;
    type RecordOutput = Result<ParquetData>;
    type EncodeOutput = Result<(RecordBatch, Arc<Schema>)>;

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
        let encoded_seq = self.encoder_seq(seq, self.option.kmer_size);
        let encoded_seq_str: Vec<String> = encoded_seq
            .into_par_iter()
            .map(|x| String::from_utf8_lossy(x).to_string())
            .collect();

        // encode the quality
        let (encoded_qual, encoded_kmer_qual) =
            self.encode_qual(qual, self.option.kmer_size, self.option.qual_offset);
        let encoded_target = self.encode_target(id, Some(encoded_seq_str.len()))?;
        let result = ParquetDataBuilder::default()
            .id(String::from_utf8_lossy(id).to_string())
            .kmer_seq(encoded_seq_str)
            .kmer_qual(encoded_kmer_qual)
            .kmer_target(encoded_target)
            .qual(encoded_qual)
            .build()
            .context("Failed to build parquet data")?;
        Ok(result)
    }

    fn encode<P: AsRef<Path>>(&mut self, path: P) -> Self::EncodeOutput {
        // Define the schema of the data (one column of integers)
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Utf8, false),
            Field::new(
                "kmer_seq",
                DataType::List(Box::new(Field::new("item", DataType::Utf8, true)).into()),
                false,
            ),
            Field::new(
                "kmer_qual",
                DataType::List(Box::new(Field::new("item", DataType::Int32, true)).into()),
                false,
            ),
            Field::new(
                "kmer_target",
                DataType::List(Box::new(Field::new("item", DataType::Int32, true)).into()),
                false,
            ),
            Field::new(
                "qual",
                DataType::List(Box::new(Field::new("item", DataType::Int32, true)).into()),
                false,
            ),
        ]));

        let records = self.fetch_records(path, self.option.kmer_size)?;

        let data: Vec<ParquetData> = records
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

        // Create builders for each field
        let mut id_builder = StringBuilder::new();
        let mut kmer_seq_builder = ListBuilder::new(StringBuilder::new());
        let mut kmer_qual_builder = ListBuilder::new(Int32Builder::new());
        let mut kmer_target_builder = ListBuilder::new(Int32Builder::new());
        let mut qual_builder = ListBuilder::new(Int32Builder::new());

        // Populate builders
        data.into_iter().for_each(|parquet_record| {
            id_builder.append_value(&parquet_record.id);

            parquet_record.kmer_seq.into_iter().for_each(|seq| {
                kmer_seq_builder.values().append_value(seq);
            });
            kmer_seq_builder.append(true); // Finish the current list item

            parquet_record.qual.into_iter().for_each(|qual| {
                qual_builder.values().append_value(qual);
            });
            qual_builder.append(true);

            parquet_record.kmer_qual.into_iter().for_each(|kmer_qual| {
                kmer_qual_builder.values().append_value(kmer_qual);
            });
            kmer_qual_builder.append(true);

            parquet_record
                .kmer_target
                .into_iter()
                .for_each(|kmer_target| {
                    kmer_target_builder.values().append_value(kmer_target);
                });
            kmer_target_builder.append(true);
        });

        // Build arrays
        let id_array = Arc::new(id_builder.finish());
        let kmer_seq_array = Arc::new(kmer_seq_builder.finish());
        let kmer_qual_array = Arc::new(kmer_qual_builder.finish());
        let qual_array = Arc::new(qual_builder.finish());
        let kmer_target_array = Arc::new(kmer_target_builder.finish());

        // Create a RecordBatch
        let record_batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                id_array as Arc<dyn Array>,
                kmer_seq_array as Arc<dyn Array>,
                kmer_qual_array as Arc<dyn Array>,
                kmer_target_array as Arc<dyn Array>,
                qual_array as Arc<dyn Array>,
            ],
        )?;

        Ok((record_batch, schema))
    }

    fn encode_multiple(&mut self, paths: &[PathBuf], parallel: bool) -> Self::EncodeOutput {
        // let result = if parallel {
        //     paths
        //         .into_par_iter()
        //         .map(|path| {
        //             let mut encoder = self.clone();
        //             encoder.encode(path)
        //         })
        //         .collect::<Result<Vec<_>>>()?
        // } else {
        //     paths
        //         .iter()
        //         .map(|path| {
        //             let mut encoder = self.clone();
        //             encoder.encode(path)
        //         })
        //         .collect::<Result<Vec<_>>>()?
        // };
        todo!()
    }
}

#[cfg(test)]
mod tests {
    use crate::fq_encode::FqEncoderOptionBuilder;

    use super::*;
    #[test]
    fn test_encode_fq_for_parquet() {
        let option = FqEncoderOptionBuilder::default()
            .kmer_size(3)
            .vectorized_target(false)
            .build()
            .unwrap();

        let mut encoder = ParquetEncoderBuilder::default()
            .option(option)
            .build()
            .unwrap();
        let (record_batch, scheme) = encoder.encode("tests/data/twenty_five_records.fq").unwrap();
        // write_parquet("test.parquet", record_batch, scheme).unwrap();
    }
}
