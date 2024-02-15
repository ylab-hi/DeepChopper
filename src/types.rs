use ndarray::{Array2, Array3};
use std::collections::HashMap;

pub type Element = i64;
pub type Matrix = Array2<Element>;
pub type Tensor = Array3<Element>;
pub type Kmer2IdTable = HashMap<Vec<u8>, Element>;
pub type Id2KmerTable = HashMap<Element, Vec<u8>>;
