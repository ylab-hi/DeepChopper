use anyhow::Result;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

pub fn write_json<P: AsRef<Path>>(path: P, values: Vec<serde_json::Value>) -> Result<()> {
    let output = File::create(path.as_ref()).unwrap();
    let mut buf = BufWriter::new(output);
    values.into_iter().for_each(|record| {
        buf.write_all(record.to_string().as_bytes()).unwrap();
    });
    Ok(())
}
