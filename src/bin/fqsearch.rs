use anyhow::Result;
use clap::Parser;
use rayon::prelude::*;
use std::path::{Path, PathBuf};

use deepchopper::output;
use noodles::fastq;

#[derive(Parser, Debug)]
#[command(version, about = "Fuzzy search id from fq", long_about = None)]
struct Cli {
    /// path to the integrated fq file
    fq: PathBuf,

    /// needle to search
    #[arg(long, value_name = "id", conflicts_with = "id_file")]
    id: Option<String>,

    /// needles file to search
    #[arg(long, value_name = "file", conflicts_with = "id")]
    id_file: Option<PathBuf>,

    /// threads number
    #[arg(short, long, default_value = "2")]
    threads: Option<usize>,

    /// Turn debugging information on
    #[arg(short, long, action = clap::ArgAction::Count)]
    debug: u8,
}

fn find_needle<P: AsRef<Path>>(haystack: P, needles: &[String]) -> Result<Vec<fastq::Record>> {
    let records = output::read_noodel_records_from_fq_or_zip_fq(haystack)?;
    Ok(records
        .into_par_iter()
        .filter(|record| {
            let id = String::from_utf8(record.definition().name().to_vec()).unwrap();
            needles.par_iter().any(|needle| id.contains(needle))
        })
        .collect())
}

fn show_records(records: &[fastq::Record]) {
    let handle = std::io::stdout().lock();
    let mut writer = fastq::io::Writer::new(handle);
    for record in records {
        writer.write_record(record).unwrap();
    }
}

fn main() -> Result<()> {
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

    if cli.id.is_none() && cli.id_file.is_none() {
        log::error!("Please provide either --id or --id-file");
        // exit with error
        std::process::exit(1);
    }

    let select_ids = if let Some(id) = cli.id {
        vec![id]
    } else {
        let ids = std::fs::read_to_string(cli.id_file.unwrap())?;
        ids.par_lines().map(|s| s.to_string()).collect()
    };


    let records = find_needle(cli.fq, &select_ids)?;
    show_records(&records);
    Ok(())
}
