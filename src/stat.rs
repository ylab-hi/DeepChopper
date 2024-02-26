use std::path::Path;

use anyhow::{Context, Result};
use needletail::parse_fastx_file;
use noodles::bam;
use rayon::prelude::*;

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

pub fn summary_bam_record_len<P: AsRef<Path>>(path: P) -> Result<Vec<usize>> {
    let mut reader = bam::io::reader::Builder.build_from_path(path.as_ref())?;
    let _header = reader.read_header()?;
    let result: Vec<usize> = reader
        .records()
        .par_bridge()
        .map(|res| {
            let record = res.context("valid record")?;
            let seq_len = record.sequence().len();
            Ok(seq_len)
        })
        .collect::<Result<Vec<usize>>>()?;

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_summary_bam_record_len() {
        let path = "tests/data/reads.bam";
        let mut result = summary_bam_record_len(path).unwrap();
        let mut expect = vec![3863, 4041, 3739, 4041, 3863, 3739];
        result.par_sort();
        expect.par_sort();

        assert_eq!(result, expect);
    }
}
