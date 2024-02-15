use std::io;

use anyhow::Result;
use noodles::fastq::{self as fastq, record::Definition};

use crate::fq_encode::RecordData;

pub fn write_fq(records: &[RecordData]) -> Result<()> {
    let stdout = io::stdout().lock();
    let mut writer = fastq::Writer::new(stdout);

    let record = fastq::Record::new(Definition::new("r0", ""), "ACGT", "NDLS");
    writer.write_record(&record)?;

    Ok(())
}
