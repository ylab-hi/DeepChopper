use std::path::Path;

use anyhow::{Context, Result};
use needletail::parse_fastx_file;

pub fn summary_record_len<P: AsRef<Path>>(path: P) -> Result<Vec<usize>> {
    let mut reader = parse_fastx_file(path.as_ref()).context("valid path/file")?;
    let mut result = Vec::new();

    while let Some(record) = reader.next() {
        let seqrec = record.context("invalid record")?;
        let seq_len = seqrec.num_bases();
        result.push(seq_len);
    }

    Ok(result)
}
