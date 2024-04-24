use anyhow::Result;
use noodles::bam;
use pyo3::prelude::*;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

use noodles::sam::alignment::record::cigar::op::{Kind, Op};
use noodles::sam::record::Cigar;

pub fn cigar_to_string(cigar: &[Op]) -> Result<String> {
    let mut cigar_str = String::new();
    for op in cigar {
        let kind_str = match op.kind() {
            Kind::Match => "M",
            Kind::Insertion => "I",
            Kind::Deletion => "D",
            Kind::Skip => "N",
            Kind::SoftClip => "S",
            Kind::HardClip => "H",
            Kind::Pad => "P",
            Kind::SequenceMatch => "=",
            Kind::SequenceMismatch => "X",
        };
        cigar_str.push_str(&format!("{}{}", op.len(), kind_str));
    }
    Ok(cigar_str)
}

pub fn _calc_softclips(cigars: &[Op]) -> Result<(usize, usize)> {
    let len = cigars.len();

    // Calculate leading soft clips
    let left_softclips = if len > 0 && cigars[0].kind() == Kind::SoftClip {
        cigars[0].len()
    } else if len > 1 && cigars[0].kind() == Kind::HardClip && cigars[1].kind() == Kind::SoftClip {
        cigars[1].len()
    } else {
        0
    };

    // Calculate trailing soft clips
    let right_softclips = if len > 0 && cigars[len - 1].kind() == Kind::SoftClip {
        cigars[len - 1].len()
    } else if len > 1
        && cigars[len - 1].kind() == Kind::HardClip
        && cigars[len - 2].kind() == Kind::SoftClip
    {
        cigars[len - 2].len()
    } else {
        0
    };

    Ok((left_softclips, right_softclips))
}

pub fn calc_softclips(cigar: &Cigar) -> Result<(usize, usize)> {
    let ops: Vec<Op> = cigar.iter().collect::<Result<Vec<_>, _>>()?;
    _calc_softclips(&ops)
}

#[pyfunction]
pub fn left_right_soft_clip(cigar_string: &str) -> Result<(usize, usize)> {
    let cigar = Cigar::new(cigar_string.as_bytes());
    calc_softclips(&cigar)
}

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

use noodles::bgzf;
use std::{fs::File, num::NonZeroUsize, thread};

pub fn read_bam_records_parallel<P: AsRef<Path>>(path: P) -> Result<HashMap<String, BamRecord>> {
    let file = File::open(path)?;

    let worker_count = thread::available_parallelism().unwrap_or(NonZeroUsize::MIN);
    let decoder = bgzf::MultithreadedReader::with_worker_count(worker_count, file);

    let mut reader = bam::io::Reader::from(decoder);
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

#[pyfunction]
#[pyo3(name = "read_bam_records_parallel")]
pub fn py_read_bam_records_parallel(path: &str) -> Result<HashMap<String, BamRecord>> {
    read_bam_records_parallel(path)
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

    #[test]
    fn test_read_bam_parallel() {
        let path = "tests/data/4reads.bam";
        let _records = read_bam_records_parallel(path).unwrap();
        println!("{:?}", _records);
    }

    #[test]
    fn test_cigar_soft_clip() {
        let (left, right) = calc_softclips(&Cigar::new(b"5S10M5S")).unwrap();
        assert_eq!(left, 5);
        assert_eq!(right, 5);

        let (left, right) = calc_softclips(&Cigar::new(b"5H10S5S")).unwrap();
        assert_eq!(left, 10);
        assert_eq!(right, 5);

        let (left, right) = calc_softclips(&Cigar::new(b"10S5M1D")).unwrap();
        assert_eq!(left, 10);
        assert_eq!(right, 0);

        let result = calc_softclips(&Cigar::new(b"1D5M10S5A"));
        assert!(result.is_err());
    }
}
