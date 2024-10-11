use anyhow::Result;
use clap::Parser;
use rayon::prelude::*;
use std::path::{Path, PathBuf};

use deepchopper::output;
use noodles::fasta;

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Cli {
    /// path to the integrated fq file
    fq: PathBuf,

    /// threads number
    #[arg(short, long, default_value = "2")]
    threads: Option<usize>,

    /// Turn debugging information on
    #[arg(short, long, action = clap::ArgAction::Count)]
    debug: u8,
}

fn covert<P: AsRef<Path>>(fq: P) -> Result<()> {
    let fq_records = output::read_noodle_records_from_bzip_fq(&fq)?;
    log::info!("converting {} records", fq_records.len());

    let handle = std::io::stdout().lock();
    let mut writer = fasta::io::Writer::new(handle);

    let fa_records: Vec<fasta::Record> = fq_records
        .par_iter()
        .map(|fq_record| {
            let definition =
                fasta::record::Definition::new(fq_record.definition().name().to_vec(), None);
            let seq = fasta::record::Sequence::from(fq_record.sequence().to_vec());
            fasta::Record::new(definition, seq)
        })
        .collect();

    for fa_record in fa_records {
        writer.write_record(&fa_record)?;
    }

    Ok(())
}

fn main() {
    let start = std::time::Instant::now();
    let cli = Cli::parse();

    let log_level = match cli.debug {
        0 => log::LevelFilter::Info,
        1 => log::LevelFilter::Debug,
        2 => log::LevelFilter::Trace,
        _ => log::LevelFilter::Trace,
    };
    // set log level
    env_logger::builder().filter_level(log_level).init();

    rayon::ThreadPoolBuilder::new()
        .num_threads(cli.threads.unwrap())
        .build_global()
        .unwrap();

    covert(cli.fq).unwrap();

    let elapsed = start.elapsed();
    log::info!("elapsed time: {:.2?}", elapsed);
}
