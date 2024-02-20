use std::fs::File;
use std::path::Path;
use std::sync::Arc;

use anyhow::Result;
use arrow::datatypes::Schema;
use arrow::record_batch::RecordBatch;
use parquet::arrow::arrow_writer::ArrowWriter;
use parquet::file::properties::WriterProperties;

pub fn write_parquet<P: AsRef<Path>>(
    path: P,
    record_batch: RecordBatch,
    schema: Arc<Schema>,
) -> Result<()> {
    let file = File::create(path.as_ref())?;
    let props = WriterProperties::builder().build();
    let mut writer = ArrowWriter::try_new(file, schema, Some(props))?;
    writer.write(&record_batch)?;
    writer.close()?;
    Ok(())
}
