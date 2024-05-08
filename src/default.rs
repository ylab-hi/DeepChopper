pub const QUAL_OFFSET: u8 = 33;
pub const BASES: &[u8] = b"ATCGN";
pub const KMER_SIZE: u8 = 3;
pub const VECTORIZED_TARGET: bool = false;
pub const MIN_READ_LEN: usize = 150; // 150 may be better
pub const MIN_CHOPED_SEQ_LEN: usize = 50; // 150 may be better
pub const IGNORE_LABEL: i64 = -100;
