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

impl FqEncoderOption {
    pub fn new(
        kmer_size: u8,
        qual_offset: Option<u8>,
        base: Option<&[u8]>,
        vectorized_target: Option<bool>,
    ) -> Self {
        let base = base.unwrap_or(BASES);
        let qual_offset = qual_offset.unwrap_or(QUAL_OFFSET);
        let vectorized_target = vectorized_target.unwrap_or(VECTORIZED_TARGET);

        Self {
            kmer_size,
            qual_offset,
            bases: base.to_vec(),
            vectorized_target,
        }
    }
}
