use crate::{
    default::{BASES, KMER_SIZE, QUAL_OFFSET, VECTORIZED_TARGET},
    fq_encode, kmer, output,
    types::{Element, Id2KmerTable, Kmer2IdTable},
};
use anyhow::Context;
use numpy::{IntoPyArray, PyArray2, PyArray3};
use pyo3::prelude::*;
use rayon::prelude::*;
use std::{collections::HashMap, path::PathBuf};

#[pymodule]
fn default(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add("QUAL_OFFSET", QUAL_OFFSET)?;
    m.add("BASES", BASES)?;
    m.add("KMER_SIZE", KMER_SIZE)?;
    m.add("VECTORIZED_TARGET", VECTORIZED_TARGET)?;
    Ok(())
}

#[pyclass]
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
fn write_fq(records_data: Vec<PyRecordData>, file_path: Option<PathBuf>) -> PyResult<()> {
    let records: Vec<fq_encode::RecordData> = records_data
        .into_par_iter()
        .map(|py_record| py_record.0)
        .collect();

    output::write_fq(&records, file_path)
        .context("failed to write fastq")
        .unwrap();
    Ok(())
}

#[pyfunction]
fn write_fq_parallel(
    records_data: Vec<PyRecordData>,
    file_path: PathBuf,
    threads: usize,
) -> PyResult<()> {
    let records: Vec<fq_encode::RecordData> = records_data
        .into_par_iter()
        .map(|py_record| py_record.0)
        .collect();

    output::write_fq_parallel(&records, file_path, Some(threads))
        .context("failed to write fastq")
        .unwrap();
    Ok(())
}

#[pyfunction]
fn kmerids_to_seq(kmer_ids: Vec<Element>, kmer2id_table: Id2KmerTable) -> PyResult<String> {
    let result = kmer::kmerids_to_seq(&kmer_ids, kmer2id_table).map_err(|e| e.into());
    match result {
        Ok(r) => Ok(String::from_utf8_lossy(&r).to_string()),
        Err(e) => Err(e),
    }
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
) -> PyResult<(usize, usize)> {
    let result = kmer::to_kmer_target_region(&(start..end), k, seq_len).map_err(|e| e.into());
    match result {
        Ok(r) => Ok((r.start, r.end)),
        Err(e) => Err(e),
    }
}

#[pyfunction]
fn seq_to_kmers(seq: String, k: usize) -> Vec<String> {
    kmer::seq_to_kmers(seq.as_bytes(), k as u8)
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
fn encode_fq_paths(
    py: Python,
    fq_paths: Vec<PathBuf>,
    k: usize,
    bases: String,
    qual_offset: usize,
    max_width: Option<usize>,
    max_seq_len: Option<usize>,
) -> (
    &PyArray3<Element>,
    &PyArray3<Element>,
    &PyArray2<Element>,
    HashMap<String, Element>,
) {
    let option = fq_encode::FqEncoderOptionBuilder::default()
        .kmer_size(k as u8)
        .bases(bases.as_bytes().to_vec())
        .qual_offset(qual_offset as u8)
        .max_width(max_width.unwrap_or(0))
        .max_seq_len(max_seq_len.unwrap_or(0))
        .build()
        .unwrap();

    let encoder = fq_encode::FqEncoder::new(option);
    let ((input, target), qual) = encoder.encode_fq_paths(&fq_paths).unwrap();

    let kmer2id: HashMap<String, Element> = encoder
        .kmer2id_table
        .par_iter()
        .map(|(k, v)| (String::from_utf8_lossy(k).to_string(), *v))
        .collect();

    (
        input.into_pyarray(py),
        target.into_pyarray(py),
        qual.into_pyarray(py),
        kmer2id,
    )
}

#[pyfunction]
fn encode_fq_path(
    py: Python,
    fq_path: PathBuf,
    k: usize,
    bases: String,
    qual_offset: usize,
    vectorized_target: bool,
    max_width: Option<usize>,
    max_seq_len: Option<usize>,
) -> (
    &PyArray3<Element>,
    &PyArray3<Element>,
    &PyArray2<Element>,
    HashMap<String, Element>,
) {
    let option = fq_encode::FqEncoderOptionBuilder::default()
        .kmer_size(k as u8)
        .bases(bases.as_bytes().to_vec())
        .qual_offset(qual_offset as u8)
        .max_width(max_width.unwrap_or(0))
        .max_seq_len(max_seq_len.unwrap_or(0))
        .vectorized_target(vectorized_target)
        .build()
        .unwrap();

    let mut encoder = fq_encode::FqEncoder::new(option);
    let ((input, target), qual) = encoder.encode_fq_path(fq_path).unwrap();

    let kmer2id: HashMap<String, Element> = encoder
        .kmer2id_table
        .par_iter()
        .map(|(k, v)| (String::from_utf8_lossy(k).to_string(), *v))
        .collect();

    (
        input.into_pyarray(py),
        target.into_pyarray(py),
        qual.into_pyarray(py),
        kmer2id,
    )
}

#[pyfunction]
fn test_string() -> PyResult<String> {
    Ok("Hello from Rust!".to_string())
}

/// A Python module implemented in Rust.
#[pymodule]
fn deepchopper(_py: Python, m: &PyModule) -> PyResult<()> {
    pyo3_log::init();

    let default_module = PyModule::new(_py, "default")?;
    default(_py, default_module)?;
    m.add_submodule(default_module)?;

    m.add_function(wrap_pyfunction!(test_string, m)?)?;
    m.add_function(wrap_pyfunction!(seq_to_kmers, m)?)?;
    m.add_function(wrap_pyfunction!(kmers_to_seq, m)?)?;
    m.add_function(wrap_pyfunction!(generate_kmers_table, m)?)?;
    m.add_function(wrap_pyfunction!(generate_kmers, m)?)?;
    m.add_function(wrap_pyfunction!(encode_fq_path, m)?)?;
    m.add_function(wrap_pyfunction!(to_kmer_target_region, m)?)?;
    m.add_function(wrap_pyfunction!(to_original_targtet_region, m)?)?;
    m.add_function(wrap_pyfunction!(kmerids_to_seq, m)?)?;
    m.add_function(wrap_pyfunction!(write_fq, m)?)?;
    m.add_function(wrap_pyfunction!(write_fq_parallel, m)?)?;
    m.add_function(wrap_pyfunction!(encode_fq_paths, m)?)?;

    m.add_class::<PyRecordData>()?;

    Ok(())
}
