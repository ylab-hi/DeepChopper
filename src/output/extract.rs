use std::path::Path;

use anyhow::{Context, Result};
use bstr::BString;
use needletail::parse_fastx_file;

use crate::fq_encode::RecordData;

pub fn extract_records_by_ids<P: AsRef<Path>>(ids: &[BString], path: P) -> Result<Vec<RecordData>> {
    let mut reader = parse_fastx_file(path.as_ref()).context("valid path/file")?;
    let mut records: Vec<RecordData> = Vec::new();

    while let Some(record) = reader.next() {
        let seqrec = record.context("invalid record")?;
        let id = BString::new(seqrec.id().to_vec());

        if !ids.contains(&id) {
            continue;
        }

        let seq = seqrec.seq();
        let qual = seqrec.qual().context("invalid qual")?;
        records.push((id.to_vec(), seq.to_vec(), qual.to_vec()).into());
    }
    Ok(records)
}
