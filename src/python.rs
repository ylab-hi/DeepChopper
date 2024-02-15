use crate::{
    fq_encode::{self, Element},
    kmer::{self},
};
use numpy::{IntoPyArray, PyArray3};
use pyo3::prelude::*;
use rayon::prelude::*;
use std::{collections::HashMap, path::PathBuf};

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
fn generate_kmers_table(base: String, k: usize) -> kmer::KmerTable {
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
fn encode_fqs(
    py: Python,
    fq_path: PathBuf,
    k: usize,
    bases: String,
    qual_offset: usize,
) -> (
    &PyArray3<Element>,
    &PyArray3<Element>,
    HashMap<String, Element>,
) {
    let option = fq_encode::FqEncoderOptionBuilder::default()
        .kmer_size(k as u8)
        .bases(bases.as_bytes().to_vec())
        .qual_offset(qual_offset as u8)
        .build()
        .unwrap();

    let encoder = fq_encode::FqEncoder::new(option);
    let (input, target) = encoder.encoder_fqs(fq_path).unwrap();

    let kmer2id: HashMap<String, Element> = encoder
        .kmer_table
        .par_iter()
        .map(|(k, v)| (String::from_utf8_lossy(k).to_string(), *v))
        .collect();

    (input.into_pyarray(py), target.into_pyarray(py), kmer2id)
}

#[pyfunction]
fn test_string() -> PyResult<String> {
    Ok("Hello from Rust!".to_string())
}

/// A Python module implemented in Rust.
#[pymodule]
fn deepchopper(_py: Python, m: &PyModule) -> PyResult<()> {
    pyo3_log::init();

    m.add_function(wrap_pyfunction!(test_string, m)?)?;
    m.add_function(wrap_pyfunction!(seq_to_kmers, m)?)?;
    m.add_function(wrap_pyfunction!(kmers_to_seq, m)?)?;
    m.add_function(wrap_pyfunction!(generate_kmers_table, m)?)?;
    m.add_function(wrap_pyfunction!(generate_kmers, m)?)?;
    m.add_function(wrap_pyfunction!(encode_fqs, m)?)?;
    Ok(())
}
