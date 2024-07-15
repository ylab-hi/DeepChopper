use std::{
    fmt::Display,
    path::{Path, PathBuf},
    sync::Arc,
};

use arrow::array::{Array, Int32Builder, ListBuilder, RecordBatch, StringBuilder};
use arrow::datatypes::{DataType, Field, Schema};

use bstr::BString;
use derive_builder::Builder;
use log::info;
use serde_derive::{Deserialize, Serialize};

use crate::{output::write_parquet, types::Element};

use super::{triat::Encoder, FqEncoderOption, RecordData};
use anyhow::{Context, Result};
use pyo3::prelude::*;
use rayon::prelude::*;

#[derive(Debug, Builder, Default)]
pub struct ParquetData {
    pub id: BString,          // id
    pub seq: BString,         // kmer_seq
    pub qual: Vec<Element>,   // kmer_qual
    pub target: Vec<Element>, // kmer_target
}

#[pyclass]
#[derive(Debug, Builder, Default, Clone, Serialize, Deserialize)]
pub struct ParquetEncoder {
    pub option: FqEncoderOption,
}

impl ParquetEncoder {
    pub fn new(option: FqEncoderOption) -> Self {
        Self { option }
    }

    fn generate_schema(&self) -> Arc<Schema> {
        Arc::new(Schema::new(vec![
            Field::new("id", DataType::Utf8, false),
            Field::new("seq", DataType::Utf8, false),
            Field::new(
                "qual",
                DataType::List(Box::new(Field::new("item", DataType::Int32, true)).into()),
                false,
            ),
            Field::new(
                "target",
                DataType::List(Box::new(Field::new("item", DataType::Int32, true)).into()),
                false,
            ),
        ]))
    }

    fn generate_batch(&self, records: &[RecordData], schema: &Arc<Schema>) -> Result<RecordBatch> {
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
        let mut seq_builder = StringBuilder::new();
        let mut qual_builder = ListBuilder::new(Int32Builder::new());
        let mut target_builder = ListBuilder::new(Int32Builder::new());

        // Populate builders
        data.into_iter().for_each(|parquet_record| {
            id_builder.append_value(parquet_record.id.to_string());
            seq_builder.append_value(parquet_record.seq.to_string());

            parquet_record.qual.into_iter().for_each(|qual| {
                qual_builder.values().append_value(qual);
            });
            qual_builder.append(true);

            parquet_record.target.into_iter().for_each(|target| {
                target_builder.values().append_value(target);
            });
            target_builder.append(true);
        });

        // Build arrays
        let id_array = Arc::new(id_builder.finish());
        let seq_array = Arc::new(seq_builder.finish());
        let qual_array = Arc::new(qual_builder.finish());
        let target_array = Arc::new(target_builder.finish());

        // Create a RecordBatch
        let record_batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                id_array as Arc<dyn Array>,
                seq_array as Arc<dyn Array>,
                qual_array as Arc<dyn Array>,
                target_array as Arc<dyn Array>,
            ],
        )?;
        Ok(record_batch)
    }

    pub fn encode_chunk<P: AsRef<Path>>(
        &mut self,
        path: P,
        chunk_size: usize,
        parallel: bool,
    ) -> Result<()> {
        let schema = self.generate_schema();
        let records = self.fetch_records(&path, self.option.kmer_size)?;
        info!("Encoding records with chunk size {} ", chunk_size);

        // create a folder for the chunk parquet files
        let file_name = path.as_ref().file_name().unwrap().to_str().unwrap();
        let chunks_folder = path
            .as_ref()
            .parent()
            .unwrap()
            .join(format!("{}_{}", file_name, "chunks"))
            .to_path_buf();
        // create the folder
        std::fs::create_dir_all(&chunks_folder).context("Failed to create folder for chunks")?;

        if parallel {
            records
                // .chunks(chunk_size)
                .par_chunks(chunk_size)
                .enumerate()
                .for_each(|(idx, record)| {
                    let record_batch = self
                        .generate_batch(record, &schema)
                        .context(format!("Failed to generate record batch for chunk {}", idx))
                        .unwrap();
                    let parquet_path = chunks_folder.join(format!("{}_{}.parquet", file_name, idx));
                    write_parquet(parquet_path, record_batch, schema.clone())
                        .context(format!("Failed to write parquet file for chunk {}", idx))
                        .unwrap();
                });
        } else {
            records
                .chunks(chunk_size)
                .enumerate()
                .for_each(|(idx, record)| {
                    let record_batch = self
                        .generate_batch(record, &schema)
                        .context(format!("Failed to generate record batch for chunk {}", idx))
                        .unwrap();
                    let parquet_path = chunks_folder.join(format!("{}_{}.parquet", file_name, idx));
                    write_parquet(parquet_path, record_batch, schema.clone())
                        .context(format!("Failed to write parquet file for chunk {}", idx))
                        .unwrap();
                });
        }

        Ok(())
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

    fn encode_qual(
        &self,
        qual: &[u8],
        _kmer_size: u8,
        qual_offset: u8,
    ) -> (Vec<Element>, Vec<Element>) {
        // input is quality of fastq
        // 1. convert the quality to a score
        // 2. return the score
        let encoded_qual: Vec<Element> = qual
            .par_iter()
            .map(|&q| {
                // Convert ASCII to Phred score for Phred+33 encoding
                (q - qual_offset) as Element
            })
            .collect();
        let empty: Vec<Element> = vec![];
        (encoded_qual, empty)
    }

    fn encode_target(&self, id: &[u8], _kmer_seq_len: Option<usize>) -> Self::TargetOutput {
        let target = Self::parse_target_from_id(id).context("Failed to parse target from ID")?;
        let result = target
            .into_par_iter()
            .map(|x| [x.start as Element, x.end as Element])
            .flatten()
            .collect();
        Ok(result)
    }

    fn encode_record(&self, id: &[u8], seq: &[u8], qual: &[u8]) -> Self::RecordOutput {
        // encode the quality
        let (encoded_qual, _encoded_kmer_qual) =
            self.encode_qual(qual, self.option.kmer_size, self.option.qual_offset);

        let encoded_target = self.encode_target(id, Some(seq.len()))?;

        let result = ParquetDataBuilder::default()
            .id(id.into())
            .seq(seq.into())
            .qual(encoded_qual)
            .target(encoded_target)
            .build()
            .context("Failed to build parquet data")?;
        Ok(result)
    }

    fn encode<P: AsRef<Path>>(&mut self, path: P) -> Self::EncodeOutput {
        // Define the schema of the data (one column of integers)
        let schema = self.generate_schema();
        let records = self.fetch_records(path, self.option.kmer_size)?;
        let record_batch = self.generate_batch(&records, &schema)?;
        Ok((record_batch, schema))
    }

    fn encode_multiple(&mut self, _paths: &[PathBuf], _parallel: bool) -> Self::EncodeOutput {
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
        let (record_batch, scheme) = encoder.encode("tests/data/one_record.fq").unwrap();
        write_parquet("test.parquet", record_batch, scheme).unwrap();
    }
}
