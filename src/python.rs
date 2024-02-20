use crate::{
    default::{BASES, KMER_SIZE, QUAL_OFFSET, VECTORIZED_TARGET},
    fq_encode::{self, Encoder},
    kmer,
    output::{self, write_json, write_parquet},
    types::{Element, Id2KmerTable, Kmer2IdTable},
};
use anyhow::Result;
use bstr::BString;
use needletail::Sequence;
use numpy::{IntoPyArray, PyArray2, PyArray3};
use pyo3::prelude::*;
use rayon::prelude::*;
use std::{collections::HashMap, path::PathBuf};

use log::{debug, error, info, warn};

#[pyfunction]
fn test_log() {
    debug!("debug Hello from Rust!");
    info!("info Hello from Rust!");
    warn!("warn Hello from Rust!");
    error!("error Hello from Rust!");
}

#[pymodule]
fn default(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add("QUAL_OFFSET", QUAL_OFFSET)?;
    m.add("BASES", BASES)?;
    m.add("KMER_SIZE", KMER_SIZE)?;
    m.add("VECTORIZED_TARGET", VECTORIZED_TARGET)?;
    Ok(())
}

#[pymethods]
impl fq_encode::TensorEncoder {
    #[new]
    fn py_new(
        option: fq_encode::FqEncoderOption,
        tensor_max_width: Option<usize>,
        tensor_max_seq_len: Option<usize>,
    ) -> Self {
        fq_encode::TensorEncoder::new(option, tensor_max_width, tensor_max_seq_len)
    }
}

#[pymethods]
impl fq_encode::JsonEncoder {
    #[new]
    fn py_new(option: fq_encode::FqEncoderOption) -> Self {
        fq_encode::JsonEncoder::new(option)
    }
}

#[pymethods]
impl fq_encode::ParquetEncoder {
    #[new]
    fn py_new(option: fq_encode::FqEncoderOption) -> Self {
        fq_encode::ParquetEncoder::new(option)
    }
}

#[pyfunction]
fn summary_record_len(path: PathBuf) -> Result<Vec<usize>> {
    fq_encode::summary_record_len(path)
}

#[pyclass(name = "RecordData")]
struct PyRecordData(fq_encode::RecordData);

impl From<fq_encode::RecordData> for PyRecordData {
    fn from(data: fq_encode::RecordData) -> Self {
        Self(data)
    }
}

// Implement FromPyObject for PyRecordData
impl<'source> FromPyObject<'source> for PyRecordData {
    fn extract(obj: &'source PyAny) -> PyResult<Self> {
        // Example extraction logic:
        // Assuming Python objects are tuples of (id, seq, qual)
        let (id, seq, qual): (&str, &str, &str) = obj.extract()?;
        Ok(PyRecordData(fq_encode::RecordData {
            id: id.into(),
            seq: seq.into(),
            qual: qual.into(),
        }))
    }
}

#[pymethods]
impl PyRecordData {
    #[new]
    fn new(id: String, seq: String, qual: String) -> Self {
        Self(fq_encode::RecordData {
            id: id.into(),
            seq: seq.into(),
            qual: qual.into(),
        })
    }

    #[getter]
    fn id(&self) -> String {
        self.0.id.to_string()
    }

    #[setter]
    fn set_id(&mut self, id: String) {
        self.0.id = id.into();
    }

    #[getter]
    fn seq(&self) -> String {
        self.0.seq.to_string()
    }

    #[setter]
    fn set_seq(&mut self, seq: String) {
        self.0.seq = seq.into();
    }

    #[getter]
    fn qual(&self) -> String {
        self.0.qual.to_string()
    }
    #[setter]
    fn set_qual(&mut self, qual: String) {
        self.0.qual = qual.into();
    }
}

#[pyfunction]
fn extract_records_by_ids(ids: Vec<String>, path: PathBuf) -> Result<Vec<PyRecordData>> {
    let ids: Vec<BString> = ids.into_par_iter().map(|id| id.into()).collect();

    output::extract_records_by_ids(&ids, path).map(|records| {
        records
            .into_par_iter()
            .map(|record| record.into())
            .collect()
    })
}

#[pyfunction]
fn write_fq(records_data: Vec<PyRecordData>, file_path: Option<PathBuf>) -> Result<()> {
    let records: Vec<fq_encode::RecordData> = records_data
        .into_par_iter()
        .map(|py_record| py_record.0)
        .collect();
    output::write_fq(&records, file_path)
}

#[pyfunction]
fn write_fq_parallel(
    records_data: Vec<PyRecordData>,
    file_path: PathBuf,
    threads: usize,
) -> Result<()> {
    let records: Vec<fq_encode::RecordData> = records_data
        .into_par_iter()
        .map(|py_record| py_record.0)
        .collect();

    output::write_fq_parallel(&records, file_path, Some(threads))
}

#[pyfunction]
fn kmerids_to_seq(
    kmer_ids: Vec<Element>,
    id2kmer_table: HashMap<Element, String>,
) -> Result<String> {
    // let kmer
    let id2kmer_table: Id2KmerTable = id2kmer_table
        .par_iter()
        .map(|(k, v)| (*k, v.as_bytes().to_vec()))
        .collect();
    kmer::kmerids_to_seq(&kmer_ids, id2kmer_table).map(|x| String::from_utf8_lossy(&x).to_string())
}

#[pyfunction]
fn to_original_targtet_region(start: usize, end: usize, k: usize) -> (usize, usize) {
    let original = kmer::to_original_targtet_region(&(start..end), k);
    (original.start, original.end)
}

#[pyfunction]
fn to_kmer_target_region(
    start: usize,
    end: usize,
    k: usize,
    seq_len: Option<usize>,
) -> Result<(usize, usize)> {
    kmer::to_kmer_target_region(&(start..end), k, seq_len).map(|r| (r.start, r.end))
}

#[pyfunction]
fn seq_to_kmers(seq: String, k: usize, overlap: bool) -> Vec<String> {
    let normalized_seq = seq.as_bytes().normalize(false);
    kmer::seq_to_kmers(&normalized_seq, k, overlap)
        .par_iter()
        .map(|s| String::from_utf8_lossy(s).to_string())
        .collect()
}

#[pyfunction]
fn kmers_to_seq(kmers: Vec<String>) -> String {
    let kmers_as_bytes: Vec<&[u8]> = kmers.par_iter().map(|s| s.as_bytes()).collect();
    String::from_utf8_lossy(&kmer::kmers_to_seq(kmers_as_bytes)).to_string()
}

#[pyfunction]
fn generate_kmers_table(base: String, k: usize) -> Kmer2IdTable {
    let base = base.as_bytes();
    kmer::generate_kmers_table(base, k as u8)
}

#[pyfunction]
fn generate_kmers(base: String, k: usize) -> Vec<String> {
    let base = base.as_bytes();
    kmer::generate_kmers(base, k as u8)
        .into_iter()
        .map(|s| String::from_utf8_lossy(&s).to_string())
        .collect()
}

#[pyfunction]
fn normalize_seq(seq: String, iupac: bool) -> String {
    String::from_utf8_lossy(&seq.as_bytes().normalize(iupac)).to_string()
}

#[pyfunction]
fn encode_fq_paths_to_tensor(
    py: Python,
    fq_paths: Vec<PathBuf>,
    k: usize,
    bases: String,
    qual_offset: usize,
    vectorized_target: bool,
    parallel_for_files: bool,
    max_width: Option<usize>,
    max_seq_len: Option<usize>,
) -> Result<(
    &PyArray3<Element>,
    &PyArray3<Element>,
    &PyArray2<Element>,
    HashMap<String, Element>,
)> {
    let option = fq_encode::FqEncoderOptionBuilder::default()
        .kmer_size(k as u8)
        .bases(bases.as_bytes().to_vec())
        .qual_offset(qual_offset as u8)
        .vectorized_target(vectorized_target)
        .build()?;

    let mut encoder = fq_encode::TensorEncoderBuilder::default()
        .option(option)
        .tensor_max_width(max_width.unwrap_or(0))
        .tensor_max_seq_len(max_seq_len.unwrap_or(0))
        .build()?;

    let ((input, target), qual) = encoder.encode_multiple(&fq_paths, parallel_for_files)?;

    let kmer2id: HashMap<String, Element> = encoder
        .kmer2id_table
        .par_iter()
        .map(|(k, v)| (String::from_utf8_lossy(k).to_string(), *v))
        .collect();

    Ok((
        input.into_pyarray(py),
        target.into_pyarray(py),
        qual.into_pyarray(py),
        kmer2id,
    ))
}

#[pyfunction]
fn encode_fq_path_to_tensor(
    py: Python,
    fq_path: PathBuf,
    k: usize,
    bases: String,
    qual_offset: usize,
    vectorized_target: bool,
    max_width: Option<usize>,
    max_seq_len: Option<usize>,
) -> Result<(
    &PyArray3<Element>,
    &PyArray3<Element>,
    &PyArray2<Element>,
    HashMap<String, Element>,
)> {
    let option = fq_encode::FqEncoderOptionBuilder::default()
        .kmer_size(k as u8)
        .bases(bases.as_bytes().to_vec())
        .qual_offset(qual_offset as u8)
        .vectorized_target(vectorized_target)
        .build()?;

    let mut encoder = fq_encode::TensorEncoderBuilder::default()
        .option(option)
        .tensor_max_width(max_width.unwrap_or(0))
        .tensor_max_seq_len(max_seq_len.unwrap_or(0))
        .build()?;

    let ((input, target), qual) = encoder.encode(fq_path)?;

    let kmer2id: HashMap<String, Element> = encoder
        .kmer2id_table
        .par_iter()
        .map(|(k, v)| (String::from_utf8_lossy(k).to_string(), *v))
        .collect();

    Ok((
        input.into_pyarray(py),
        target.into_pyarray(py),
        qual.into_pyarray(py),
        kmer2id,
    ))
}

#[pyfunction]
fn encode_fq_path_to_json(
    fq_path: PathBuf,
    k: usize,
    bases: String,
    qual_offset: usize,
    vectorized_target: bool,
    result_path: Option<PathBuf>,
) -> Result<()> {
    let option = fq_encode::FqEncoderOptionBuilder::default()
        .kmer_size(k as u8)
        .bases(bases.as_bytes().to_vec())
        .qual_offset(qual_offset as u8)
        .vectorized_target(vectorized_target)
        .build()?;

    let mut encoder = fq_encode::JsonEncoderBuilder::default()
        .option(option)
        .build()?;

    let result = encoder.encode(&fq_path)?;

    // result file is fq_path with .parquet extension
    let json_path = if let Some(path) = result_path {
        if path.with_extension("json").exists() {
            warn!("{} already exists, overwriting", path.display());
        }
        path.with_extension("json")
    } else {
        fq_path.with_extension("json")
    };
    write_json(json_path, result)?;
    Ok(())
}

#[pyfunction]
fn encode_fq_path_to_parquet(
    fq_path: PathBuf,
    k: usize,
    bases: String,
    qual_offset: usize,
    vectorized_target: bool,
    result_path: Option<PathBuf>,
) -> Result<()> {
    let option = fq_encode::FqEncoderOptionBuilder::default()
        .kmer_size(0)
        .bases(bases.as_bytes().to_vec())
        .qual_offset(qual_offset as u8)
        .vectorized_target(vectorized_target)
        .build()?;

    let mut encoder = fq_encode::ParquetEncoderBuilder::default()
        .option(option)
        .build()?;
    let (record_batch, schema) = encoder.encode(&fq_path)?;

    // result file is fq_path with .parquet extension
    let parquet_path = if let Some(path) = result_path {
        if path.with_extension("parquet").exists() {
            warn!("{} already exists, overwriting", path.display());
        }
        path.with_extension("parquet")
    } else {
        fq_path.with_extension("parquet")
    };
    write_parquet(parquet_path, record_batch, schema)?;
    Ok(())
}

#[pyfunction]
fn encode_fq_paths_to_parquet(
    fq_path: Vec<PathBuf>,
    k: usize,
    bases: String,
    qual_offset: usize,
    vectorized_target: bool,
) -> Result<()> {
    fq_path.iter().for_each(|path| {
        encode_fq_path_to_parquet(
            path.clone(),
            k,
            bases.clone(),
            qual_offset,
            vectorized_target,
            None,
        )
        .unwrap();
    });
    Ok(())
}

/// A Python module implemented in Rust.
#[pymodule]
fn deepchopper(_py: Python, m: &PyModule) -> PyResult<()> {
    pyo3_log::init();

    let default_module = PyModule::new(_py, "default")?;
    default(_py, default_module)?;
    m.add_submodule(default_module)?;

    m.add_function(wrap_pyfunction!(normalize_seq, m)?)?;
    m.add_function(wrap_pyfunction!(seq_to_kmers, m)?)?;
    m.add_function(wrap_pyfunction!(kmers_to_seq, m)?)?;
    m.add_function(wrap_pyfunction!(generate_kmers_table, m)?)?;
    m.add_function(wrap_pyfunction!(generate_kmers, m)?)?;
    m.add_function(wrap_pyfunction!(encode_fq_path_to_tensor, m)?)?;
    m.add_function(wrap_pyfunction!(to_kmer_target_region, m)?)?;
    m.add_function(wrap_pyfunction!(to_original_targtet_region, m)?)?;
    m.add_function(wrap_pyfunction!(kmerids_to_seq, m)?)?;
    m.add_function(wrap_pyfunction!(write_fq, m)?)?;
    m.add_function(wrap_pyfunction!(write_fq_parallel, m)?)?;
    m.add_function(wrap_pyfunction!(encode_fq_paths_to_tensor, m)?)?;
    m.add_function(wrap_pyfunction!(encode_fq_path_to_tensor, m)?)?;
    m.add_function(wrap_pyfunction!(encode_fq_path_to_parquet, m)?)?;
    m.add_function(wrap_pyfunction!(encode_fq_paths_to_parquet, m)?)?;
    m.add_function(wrap_pyfunction!(encode_fq_path_to_json, m)?)?;
    m.add_function(wrap_pyfunction!(summary_record_len, m)?)?;
    m.add_function(wrap_pyfunction!(test_log, m)?)?;
    m.add_function(wrap_pyfunction!(extract_records_by_ids, m)?)?;

    m.add_class::<PyRecordData>()?;
    m.add_class::<fq_encode::FqEncoderOption>()?;
    m.add_class::<fq_encode::TensorEncoder>()?;
    m.add_class::<fq_encode::JsonEncoder>()?;
    m.add_class::<fq_encode::ParquetEncoder>()?;

    Ok(())
}
