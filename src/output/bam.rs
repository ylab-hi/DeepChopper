use crate::utils::_calc_softclips;
use crate::utils::cigar_to_string;
use anyhow::Result;
use noodles::bam;
use pyo3::prelude::*;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

use noodles::sam::alignment::record::cigar::op::Op;

#[pyclass]
#[derive(Debug, Default, Deserialize, Serialize, FromPyObject)]
pub struct BamRecord {
    #[pyo3(get, set)]
    qname: String,
    #[pyo3(get, set)]
    mapping_quality: usize,
    #[pyo3(get, set)]
    cigar: String,
    #[pyo3(get, set)]
    forward: bool,

    #[pyo3(get, set)]
    left_softclip: usize,
    #[pyo3(get, set)]
    right_softclip: usize,
}

#[pymethods]
impl BamRecord {
    #[new]
    fn new(
        qname: String,
        mapping_quality: usize,
        cigar: String,
        forward: bool,
        left_softclip: usize,
        right_softclip: usize,
    ) -> Self {
        BamRecord {
            qname,
            mapping_quality,
            cigar,
            forward,
            left_softclip,
            right_softclip,
        }
    }
}

pub fn read_bam_records<P: AsRef<Path>>(path: P) -> Result<HashMap<String, BamRecord>> {
    let mut reader = bam::io::reader::Builder.build_from_path(path)?;
    let _header = reader.read_header()?;

    let res: Result<HashMap<String, BamRecord>> = reader
        .records()
        .par_bridge()
        .map(|result| {
            let record = result?;
            let qname = String::from_utf8(record.name().unwrap().as_bytes().to_vec())?;
            let mapping_quality = record.mapping_quality().unwrap().get() as usize;

            let ops: Vec<Op> = record.cigar().iter().collect::<Result<Vec<_>, _>>()?;
            let cigar = cigar_to_string(&ops)?;

            let forward = !record.flags().is_reverse_complemented();

            let (mut left_softclip, mut right_softclip) = _calc_softclips(&ops)?;
            if !forward {
                std::mem::swap(&mut left_softclip, &mut right_softclip);
            }

            Ok((
                qname.clone(),
                BamRecord::new(
                    qname,
                    mapping_quality,
                    cigar,
                    forward,
                    left_softclip,
                    right_softclip,
                ),
            ))
        })
        .collect();
    res
}

#[pyfunction]
#[pyo3(name = "read_bam_records")]
pub fn py_read_bam_records(path: &str) -> Result<HashMap<String, BamRecord>> {
    read_bam_records(path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_read_bam() {
        let path = "tests/data/4reads.bam";
        let _records = read_bam_records(path).unwrap();
        println!("{:?}", _records);
    }
}
