use std::path::Path;

use anyhow::{Context, Result};
use needletail::parse_fastx_file;
use noodles::bam;

pub fn summary_fx_record_len<P: AsRef<Path>>(path: P) -> Result<Vec<usize>> {
    let mut reader = parse_fastx_file(path.as_ref()).context("valid path/file")?;
    let mut result = Vec::new();

    while let Some(record) = reader.next() {
        let seqrec = record.context("invalid record")?;
        let seq_len = seqrec.num_bases();
        result.push(seq_len);
    }

    Ok(result)
}

pub fn summar_bam_record_len<P: AsRef<Path>>(path: P) -> Result<Vec<usize>> {
    let mut reader = bam::io::reader::Builder.build_from_path(path.as_ref())?;
    let mut result = Vec::new();

    for r in reader.records() {
        let record = r.context("valid record")?;
        let seq_len = record.sequence().len();
        result.push(seq_len);
    }

    Ok(result)
}
