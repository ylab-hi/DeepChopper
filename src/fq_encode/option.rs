use crate::default::{BASES, KMER_SIZE, QUAL_OFFSET, VECTORIZED_TARGET};
use derive_builder::Builder;
use serde::{Deserialize, Serialize};
use std::fmt::{self, Display, Formatter};

use pyo3::prelude::*;

#[pyclass]
#[derive(Debug, Builder, Default, Clone, Serialize, Deserialize)]
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
        threads: Option<usize>,
    ) -> Self {
        FqEncoderOptionBuilder::default()
            .kmer_size(kmer_size)
            .qual_offset(qual_offset)
            .bases(bases.as_bytes().to_vec())
            .vectorized_target(vectorized_target)
            .threads(threads.unwrap_or(2))
            .build()
            .expect("Failed to build FqEncoderOption from Python arguments.")
    }
}

impl Display for FqEncoderOption {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(
            f,
            "FqEncoderOption {{ kmer_size: {}, qual_offset: {}, bases: {:?}, vectorized_target: {}}}",
            self.kmer_size, self.qual_offset, self.bases, self.vectorized_target
        )
    }
}
