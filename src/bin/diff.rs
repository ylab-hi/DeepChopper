use anyhow::Result;
use clap::Parser;
use rayon::prelude::*;
use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};

use std::io::BufReader;

use ahash::HashMap;
use noodles::fastq::record::Record as FastqRecord;
use noodles::fastq::{self as fastq};

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Cli {
    /// path to the fq file
    #[arg(value_name = "fq")]
    fq: PathBuf,

    /// path to the chop fq file
    #[arg(value_name = "chop_fq")]
    chop_fq: PathBuf,

    /// threads number
    #[arg(short, long, default_value = "2")]
    threads: Option<usize>,

    /// Turn debugging information on
    #[arg(short, long, action = clap::ArgAction::Count)]
    debug: u8,
}

fn compare_length<P: AsRef<Path>>(fq: P, chop_fq: P) -> Result<HashMap<String, usize>> {
    let mut fq_reader = File::open(fq).map(BufReader::new).map(fastq::Reader::new)?;
    let fq_records: Result<HashMap<String, FastqRecord>> = fq_reader
        .records()
        .par_bridge()
        .map(|record| {
            let record = record?;
            let name = String::from_utf8(record.definition().name().to_vec())?;
            Ok((name, record))
        })
        .collect();
    let fq_records = fq_records?;

    let mut chop_reader = File::open(chop_fq)
        .map(BufReader::new)
        .map(fastq::Reader::new)?;
    let chop_records: Result<HashMap<String, FastqRecord>> = chop_reader
        .records()
        .par_bridge()
        .map(|record| {
            let record = record?;
            let name = String::from_utf8(record.definition().name().to_vec())?;
            Ok((name, record))
        })
        .collect();
    let chop_records = chop_records?;

    let result = fq_records
        .par_iter()
        .filter_map(|(name, fq_record)| {
            if let Some(chop_record) = chop_records.get(name) {
                if fq_record.sequence().len() != chop_record.sequence().len() {
                    let diff = fq_record
                        .sequence()
                        .len()
                        .abs_diff(chop_record.sequence().len());
                    return Some((name.clone(), diff));
                }
                // same length do not return
                None
            } else {
                log::warn!("{} not found in chop fq", name);
                return Some((name.clone(), fq_record.sequence().len()));
            }
        })
        .collect::<HashMap<String, usize>>();

    Ok(result)
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

    log::info!(
        "compare length of {} and chop fq {}",
        cli.fq.display(),
        cli.chop_fq.display()
    );

    let result = compare_length(cli.fq, cli.chop_fq).unwrap();

    // save to json file
    let json_file = "compare_len.json";
    let json = serde_json::to_string_pretty(&result).unwrap();
    let mut file = File::create(json_file).unwrap();
    let mut writer = std::io::BufWriter::new(&mut file);
    writer.write_all(json.as_bytes()).unwrap();

    let elapsed = start.elapsed();
    log::info!("elapsed time: {:.2?}", elapsed);
}
