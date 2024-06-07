use anyhow::Result;
use clap::Parser;
use noodles::bam;
use noodles::bgzf;
use rayon::prelude::*;
use std::path::{Path, PathBuf};
use std::{fs::File, num::NonZeroUsize, thread};

use noodles::sam::alignment::record::data::field::Tag;
use noodles::sam::alignment::record::data::field::Value;

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Cli {
    /// path to the bam file
    #[arg(value_name = "bam", action=clap::ArgAction::Append)]
    bam: Vec<PathBuf>,

    /// threads number
    #[arg(short, long, default_value = "2")]
    threads: Option<usize>,

    /// Turn debugging information on
    #[arg(short, long, action = clap::ArgAction::Count)]
    debug: u8,
}

fn count_chemeric_reads(bam: &Path, threads: Option<usize>) -> Result<usize> {
    let file = File::open(bam)?;

    let worker_count = if let Some(threads) = threads {
        NonZeroUsize::new(threads)
            .unwrap()
            .min(thread::available_parallelism().unwrap_or(NonZeroUsize::MIN))
    } else {
        thread::available_parallelism().unwrap_or(NonZeroUsize::MIN)
    };

    let decoder = bgzf::MultithreadedReader::with_worker_count(worker_count, file);

    let mut reader = bam::io::Reader::from(decoder);
    let _header = reader.read_header()?;

    let res = reader
        .records()
        .par_bridge()
        .filter_map(|result| {
            let record = result.unwrap();
            let is_mapped = !record.flags().is_unmapped();
            let is_not_secondary = !record.flags().is_secondary();
            let is_primary = !record.flags().is_supplementary();

            let has_sa_tag = matches!(
                record.data().get(&Tag::OTHER_ALIGNMENTS),
                Some(Ok(Value::String(_sa_string)))
            );

            if is_mapped && is_not_secondary && is_primary && has_sa_tag {
                Some(record)
            } else {
                None
            }
        })
        .count();
    Ok(res)
}

fn main() -> Result<()> {
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

    cli.bam.par_iter().for_each(|bam| {
        let res = count_chemeric_reads(bam, cli.threads).unwrap();
        log::info!("{}: {} chimeric reads", bam.display(), res);
    });

    let elapsed = start.elapsed();
    log::info!("elapsed time: {:.2?}", elapsed);
    Ok(())
}
