use crate::default::{BASES, KMER_SIZE, QUAL_OFFSET, VECTORIZED_TARGET};
use derive_builder::Builder;

#[derive(Debug, Builder, Default, Clone)]
pub struct FqEncoderOption {
    #[builder(default = "KMER_SIZE")]
    pub kmer_size: u8,
    #[builder(default = "QUAL_OFFSET")]
    pub qual_offset: u8,
    #[builder(default = "BASES.to_vec()")]
    pub bases: Vec<u8>,
    #[builder(default = "VECTORIZED_TARGET")]
    pub vectorized_target: bool,

    #[builder(default = "0")]
    pub max_width: usize, // control width of input and target tensor

    #[builder(default = "0")]
    pub max_seq_len: usize, // control width of original qual matrix
}
