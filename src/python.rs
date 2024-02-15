use crate::kmer;
use pyo3::prelude::*;
use rayon::prelude::*;

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
    kmer::generate_kmers(base, k as u8)
        .into_iter()
        .enumerate()
        .map(|(id, kmer)| (kmer, id))
        .collect()
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
fn test_string() -> PyResult<String> {
    Ok("Hello from Rust!".to_string())
}

/// A Python module implemented in Rust.
#[pymodule]
fn deepchopper(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(test_string, m)?)?;
    m.add_function(wrap_pyfunction!(seq_to_kmers, m)?)?;
    m.add_function(wrap_pyfunction!(kmers_to_seq, m)?)?;
    m.add_function(wrap_pyfunction!(generate_kmers_table, m)?)?;
    m.add_function(wrap_pyfunction!(generate_kmers, m)?)?;
    Ok(())
}
