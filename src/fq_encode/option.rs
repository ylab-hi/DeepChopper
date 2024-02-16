use std::fmt::{self, Display, Formatter};

use crate::default::{BASES, KMER_SIZE, QUAL_OFFSET, VECTORIZED_TARGET};
use derive_builder::Builder;

use pyo3::prelude::*;

#[pyclass]
#[derive(Debug, Builder, Default, Clone)]
pub struct FqEncoderOption {
    #[pyo3(get, set)]
    #[builder(default = "KMER_SIZE")]
    pub kmer_size: u8,

    #[pyo3(get, set)]
    #[builder(default = "QUAL_OFFSET")]
    pub qual_offset: u8,

    #[pyo3(get, set)]
    #[builder(default = "BASES.to_vec()")]
    pub bases: Vec<u8>,

    #[pyo3(get, set)]
    #[builder(default = "VECTORIZED_TARGET")]
    pub vectorized_target: bool,

    #[pyo3(get, set)]
    #[builder(default = "0")]
    pub max_width: usize, // control width of input and target tensor

    #[pyo3(get, set)]
    #[builder(default = "0")]
    pub max_seq_len: usize, // control width of original qual matrix

    #[pyo3(get, set)]
    #[builder(default = "2")]
    pub threads: usize,
}

#[pymethods]
impl FqEncoderOption {
    #[new]
    fn py_new(
        kmer_size: u8,
        qual_offset: u8,
        bases: String,
        vectorized_target: bool,
        max_width: Option<usize>,
        max_seq_len: Option<usize>,
        threads: Option<usize>,
    ) -> Self {
        FqEncoderOptionBuilder::default()
            .kmer_size(kmer_size)
            .qual_offset(qual_offset)
            .bases(bases.as_bytes().to_vec())
            .vectorized_target(vectorized_target)
            .max_width(max_width.unwrap_or(0))
            .max_seq_len(max_seq_len.unwrap_or(0))
            .threads(threads.unwrap_or(2))
            .build()
            .expect("Failed to build FqEncoderOption from Python arguments.")
    }
}

impl Display for FqEncoderOption {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(
            f,
            "FqEncoderOption {{ kmer_size: {}, qual_offset: {}, bases: {:?}, vectorized_target: {}, max_width: {}, max_seq_len: {} }}",
            self.kmer_size, self.qual_offset, self.bases, self.vectorized_target, self.max_width, self.max_seq_len
        )
    }
}
