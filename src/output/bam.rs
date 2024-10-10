use anyhow::Result;
use noodles::bam;
use noodles::bgzf;
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use std::ops::Deref;

use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::{fs::File, num::NonZeroUsize, thread};

use noodles::sam::alignment::record::cigar::op::{Kind, Op};
use noodles::sam::alignment::record::data::field::Tag;
use noodles::sam::alignment::record::data::field::Value;
use noodles::sam::record::Cigar;

use ahash::HashMap;

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
    pub qname: String,
    #[pyo3(get, set)]
    pub mapping_quality: usize,
    #[pyo3(get, set)]
    pub cigar: String,

    #[pyo3(get, set)]
    pub left_softclip: usize,
    #[pyo3(get, set)]
    pub right_softclip: usize,

    #[pyo3(get, set)]
    pub is_forward: bool,

    #[pyo3(get, set)]
    pub is_mapped: bool,

    #[pyo3(get, set)]
    pub is_supplementary: bool,

    #[pyo3(get, set)]
    pub is_secondary: bool,

    #[pyo3(get, set)]
    pub quality: Vec<u8>,

    #[pyo3(get, set)]
    pub sa_tag: Option<String>,
}

#[pymethods]
impl BamRecord {
    #[new]
    fn new(
        qname: String,
        mapping_quality: usize,
        cigar: String,
        left_softclip: usize,
        right_softclip: usize,
        is_forward: bool,
        is_mapped: bool,
        is_supplementary: bool,
        is_secondary: bool,
        quality: Vec<u8>,
        sa_tag: Option<String>,
    ) -> Self {
        BamRecord {
            qname,
            mapping_quality,
            cigar,
            left_softclip,
            right_softclip,
            is_forward,
            is_mapped,
            is_supplementary,
            is_secondary,
            quality,
            sa_tag,
        }
    }

    fn __getstate__(&self, py: Python) -> PyResult<PyObject> {
        // Serialize the struct to a JSON string
        let serialized = serde_json::to_string(self).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Failed to serialize: {}", e))
        })?;

        // Convert JSON string to Python bytes
        Ok(PyBytes::new_bound(py, serialized.as_bytes()).into())
    }

    fn __setstate__(&mut self, py: Python, state: PyObject) -> PyResult<()> {
        // Expect a bytes object for state
        let state_bytes: &PyBytes = state.extract(py)?;

        // Deserialize the JSON string into the current instance
        *self = serde_json::from_slice(state_bytes.as_bytes()).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Failed to deserialize: {}",
                e
            ))
        })?;
        Ok(())
    }

    fn __repr__(&self) -> String {
        format!(
            "BamRecord(qname={}, mapping_quality={}, cigar={}, left_softclip={}, right_softclip={}, is_forward={}, is_mapped={}, is_supplementary={})",
            self.qname, self.mapping_quality, self.cigar, self.left_softclip, self.right_softclip, self.is_forward, self.is_mapped, self.is_supplementary
        )
    }

    fn select_quality(&self, start: usize, end: usize) -> Vec<u8> {
        self.quality[start..end].to_vec()
    }
}

pub fn read_bam_records_parallel<P: AsRef<Path>>(
    path: P,
    threads: Option<usize>,
) -> Result<HashMap<String, BamRecord>> {
    let file = File::open(path)?;

    let worker_count = if let Some(threads) = threads {
        NonZeroUsize::new(threads)
            .unwrap()
            .min(thread::available_parallelism().unwrap_or(NonZeroUsize::MIN))
    } else {
        thread::available_parallelism().unwrap_or(NonZeroUsize::MIN)
    };

    let decoder = bgzf::MultithreadedReader::with_worker_count(worker_count, file);

    let mut reader = bam::io::Reader::from(decoder);
    let _header = reader.read_header()?;

    println!("Reading bam file with {} threads", worker_count);

    let res: Result<HashMap<String, BamRecord>> = reader
        .records()
        .par_bridge()
        .map(|result| {
            let record = result?;
            let qname = String::from_utf8(record.name().unwrap().to_vec())?;
            let mapping_quality = record.mapping_quality().unwrap().get() as usize;

            let ops: Vec<Op> = record.cigar().iter().collect::<Result<Vec<_>, _>>()?;
            let cigar = cigar_to_string(&ops)?;

            let is_forward = !record.flags().is_reverse_complemented();

            let is_mapped = !record.flags().is_unmapped();
            let is_supplementary = record.flags().is_supplementary();
            let is_secondary = record.flags().is_secondary();

            let quality = record.quality_scores().as_ref().to_vec();

            let sa_tag = if let Some(Ok(Value::String(sa_string))) =
                record.data().get(&Tag::OTHER_ALIGNMENTS)
            {
                Some(String::from_utf8(sa_string.to_vec())?)
            } else {
                None
            };

            let (mut left_softclip, mut right_softclip) = _calc_softclips(&ops)?;
            if !is_forward {
                std::mem::swap(&mut left_softclip, &mut right_softclip);
            }

            Ok((
                qname.clone(),
                BamRecord::new(
                    qname,
                    mapping_quality,
                    cigar,
                    left_softclip,
                    right_softclip,
                    is_forward,
                    is_mapped,
                    is_supplementary,
                    is_secondary,
                    quality,
                    sa_tag,
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
            let qname = String::from_utf8(record.name().unwrap().to_vec())?;
            let mapping_quality = record.mapping_quality().unwrap().get() as usize;

            let ops: Vec<Op> = record.cigar().iter().collect::<Result<Vec<_>, _>>()?;
            let cigar = cigar_to_string(&ops)?;

            let is_forward = !record.flags().is_reverse_complemented();
            let is_mapped = !record.flags().is_unmapped();
            let is_supplementary = record.flags().is_supplementary();
            let is_secondary = record.flags().is_secondary();

            let quality = record.quality_scores().as_ref().to_vec();

            let sa_tag = if let Some(Ok(Value::String(sa_string))) =
                record.data().get(&Tag::OTHER_ALIGNMENTS)
            {
                Some(String::from_utf8(sa_string.to_vec())?)
            } else {
                None
            };

            let (mut left_softclip, mut right_softclip) = _calc_softclips(&ops)?;
            if !is_forward {
                std::mem::swap(&mut left_softclip, &mut right_softclip);
            }

            Ok((
                qname.clone(),
                BamRecord::new(
                    qname,
                    mapping_quality,
                    cigar,
                    left_softclip,
                    right_softclip,
                    is_forward,
                    is_mapped,
                    is_supplementary,
                    is_secondary,
                    quality,
                    sa_tag,
                ),
            ))
        })
        .collect();
    res
}

pub fn collect_read_mapping_quality<T>(records: &[T]) -> Vec<usize>
where
    T: Deref<Target = BamRecord> + Sync,
{
    records
        .par_iter()
        .filter(|record| record.is_mapped)
        .map(|record| record.mapping_quality)
        .collect()
}

#[pyfunction]
#[pyo3(name = "collect_read_mapping_quality")]
pub fn py_collect_read_mapping_quality(records: Vec<PyRef<BamRecord>>) -> Vec<usize> {
    let records: Vec<&BamRecord> = records.iter().map(|record| record.deref()).collect();
    collect_read_mapping_quality(&records)
}

#[pyfunction]
#[pyo3(name = "read_bam_records")]
pub fn py_read_bam_records(path: &str) -> Result<HashMap<String, BamRecord>> {
    read_bam_records(path)
}

#[pyfunction]
#[pyo3(name = "read_bam_records_parallel")]
pub fn py_read_bam_records_parallel(
    path: &str,
    threads: Option<usize>,
) -> Result<HashMap<String, BamRecord>> {
    read_bam_records_parallel(path, threads)
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
        let _records = read_bam_records_parallel(path, Some(2)).unwrap();
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
