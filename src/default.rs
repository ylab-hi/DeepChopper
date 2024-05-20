pub const QUAL_OFFSET: u8 = 33;
pub const BASES: &[u8] = b"ATCGN";
pub const KMER_SIZE: u8 = 3;
pub const VECTORIZED_TARGET: bool = false;
pub const MIN_READ_LEN: usize = 150; // min read to be processed 150 may be better
pub const MIN_CHOPED_SEQ_LEN: usize = 20; // min read after processing  50 may be better
pub const IGNORE_LABEL: i64 = -100;
