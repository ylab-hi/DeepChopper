use clap::Parser;
use deepchopper::fq_encode::RecordData;
use std::fs::File;
use std::path::PathBuf;

use needletail::Sequence;
use std::io::BufReader;

use anyhow::Result;
use noodles::fastq;

#[derive(Parser)]
#[command(version, about, long_about = None)]
struct Cli {
    /// Sets a custom config file
    #[arg(value_name = "fq")]
    input: PathBuf,

    /// Turn debugging information on
    #[arg(short, long, action = clap::ArgAction::Count)]
    debug: u8,
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    let mut reader = File::open(cli.input)
        .map(BufReader::new)
        .map(fastq::Reader::new)?;

    let mut records: Vec<RecordData> = Vec::new();
    let mut record = fastq::Record::default();
    while reader.read_record(&mut record)? > 0 {
        let id = record.definition().name();
        println!("id: {:?}", String::from_utf8_lossy(id));
        let seq = record.sequence();
        let normalized_seq = seq.normalize(false);
        let qual = record.quality_scores();
        records.push((id.to_vec(), normalized_seq.to_vec(), qual.to_vec()).into());
    }

    println!("records: {:?}", records.len());
    Ok(())
}
