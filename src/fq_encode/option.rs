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
}
