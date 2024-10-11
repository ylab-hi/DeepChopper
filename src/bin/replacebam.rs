use anyhow::Result;
use clap::Parser;
use noodles::bam;
use noodles::bgzf;
use rayon::prelude::*;
use std::path::{Path, PathBuf};
use std::{fs::File, num::NonZeroUsize, thread};

use ahash::HashMap;
use ahash::HashSet;
use bstr::BString;

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Cli {
    /// path to the dc bam file
    #[arg(long)]
    dcbam: PathBuf,

    #[arg(long)]
    /// path to the do bam file
    dobam: PathBuf,

    /// threads number
    #[arg(short, long, default_value = "2")]
    threads: Option<usize>,

    /// Turn debugging information on
    #[arg(short, long, action = clap::ArgAction::Count)]
    debug: u8,
}

fn load_internal_read<P: AsRef<Path>>(
    path: P,
    threads: Option<usize>,
) -> Result<HashMap<BString, bam::record::Record>> {
    let file = File::open(path)?;

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

    let res: HashMap<BString, bam::record::Record> = reader
        .records()
        .par_bridge()
        .filter_map(|result| {
            let record = result.unwrap();
            let name = record.name().unwrap().to_vec();

            if name.contains(&b'I') {
                Some((name.into(), record))
            } else {
                None
            }
        })
        .collect();

    Ok(res)
}

fn replace_internal<P: AsRef<Path>>(dc_path: P, do_path: P, threads: Option<usize>) -> Result<()> {
    let internal_records = load_internal_read(dc_path, threads)?;
    let internal_records_names: HashSet<BString> = internal_records
        .keys()
        .par_bridge()
        .map(|key| {
            assert!(key.contains(&b'I'));
            let name = key.split(|&c| c == b'|').next().unwrap();
            name.into()
        })
        .collect();

    let file = File::open(do_path)?;
    let worker_count = if let Some(threads) = threads {
        NonZeroUsize::new(threads)
            .unwrap()
            .min(thread::available_parallelism().unwrap_or(NonZeroUsize::MIN))
    } else {
        thread::available_parallelism().unwrap_or(NonZeroUsize::MIN)
    };

    let decoder = bgzf::MultithreadedReader::with_worker_count(worker_count, file);
    let mut reader = bam::io::Reader::from(decoder);
    let header = reader.read_header()?;

    let mut res: Vec<bam::record::Record> = reader
        .records()
        .par_bridge()
        .filter_map(|result| {
            let record = result.unwrap();
            let name: BString = record.name().unwrap().to_vec().into();

            if !internal_records_names.contains(&name) {
                Some(record)
            } else {
                None
            }
        })
        .collect();

    res.extend(internal_records.into_values());

    let stdout = std::io::stdout().lock();
    let mut writer = bam::io::Writer::new(stdout);
    writer.write_header(&header)?;
    for record in res {
        writer.write_record(&header, &record)?;
    }

    Ok(())
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

    replace_internal(cli.dcbam, cli.dobam, cli.threads)?;

    let elapsed = start.elapsed();
    log::info!("elapsed time: {:.2?}", elapsed);
    Ok(())
}
