use ahash::HashSet;
use anyhow::Result;
use bstr::BString;
use clap::Parser;
use noodles::bgzf;
use rayon::prelude::*;
use std::io::BufRead;
use std::io::BufReader;
use std::io::Write;
use std::path::Path;
use std::path::PathBuf;

use std::fs::File;

use ahash::HashMap;
use noodles::bam;

use noodles::sam::alignment::record::data::field::Tag;
use noodles::sam::alignment::record::data::field::Value;
use std::{num::NonZeroUsize, thread};

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Cli {
    /// path to the compared bam
    #[arg(value_name = "bam")]
    bam: PathBuf,

    /// names of the selected reads
    #[arg(short, long)]
    names: Option<PathBuf>,

    /// threads number
    #[arg(short, long, default_value = "2")]
    threads: Option<usize>,

    /// prefix for output files
    #[arg(short, long)]
    output_prefix: Option<String>,

    /// Turn debugging information on
    #[arg(short, long, action = clap::ArgAction::Count)]
    debug: u8,
}

fn summary_sa_info<P: AsRef<Path>>(
    bam: P,
    threads: Option<usize>,
    names: Option<HashSet<String>>,
) -> Result<HashMap<String, Vec<String>>> {
    let worker_count = if let Some(threads) = threads {
        std::num::NonZeroUsize::new(threads)
            .unwrap()
            .min(thread::available_parallelism().unwrap_or(NonZeroUsize::MIN))
    } else {
        thread::available_parallelism().unwrap_or(NonZeroUsize::MIN)
    };

    let file = File::open(bam.as_ref())?;
    let decoder = bgzf::MultithreadedReader::with_worker_count(worker_count, file);
    let mut reader = bam::io::Reader::from(decoder);
    let header = reader.read_header()?;
    let references = header.reference_sequences();

    let res: HashMap<String, Vec<String>> = reader
        .records()
        .par_bridge()
        .filter_map(|result| {
            let record = result.unwrap();
            let is_mapped = !record.flags().is_unmapped();
            let is_not_secondary = !record.flags().is_secondary();
            let is_primary = !record.flags().is_supplementary();

            if is_primary && is_mapped && is_not_secondary {
                let name = String::from_utf8(record.name().unwrap().to_vec()).unwrap();

                if let Some(names_v) = &names {
                    if name.contains('|') {
                        let id = name.split('|').next().unwrap();
                        if names_v.contains(id) {
                            return Some((name, record));
                        }
                    } else if names_v.contains(&name) {
                        return Some((name, record));
                    }
                } else {
                    return Some((name, record));
                }
            }
            None
        })
        .map(|(name, record)| {
            let reference_id = record.reference_sequence_id().unwrap().unwrap();
            // get the reference name
            let reference_name = references.get_index(reference_id).unwrap().0;
            let reference_start = record.alignment_start().unwrap().unwrap();
            let alignment_info = format!("{}:{}", reference_name, reference_start);
            let mut res = vec![alignment_info];

            if let Some(Ok(Value::String(sa_string))) = record.data().get(&Tag::OTHER_ALIGNMENTS) {
                // has sa tag
                let mut splits = sa_string.split(|c| c == &b',');

                let sa_reference_name = BString::new(splits.next().unwrap().to_vec());
                let sa_start = BString::new(splits.next().unwrap().to_vec());

                let sa_alignment_info = format!("{}:{}", sa_reference_name, sa_start);
                res.push(sa_alignment_info);
            }
            (name, res)
        })
        .collect();

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

    let mut names = None;

    if let Some(names_file) = cli.names {
        log::info!("Selecting by names");

        let name_file_handle = File::open(names_file)?;
        let reader = BufReader::new(name_file_handle);
        names = Some(reader.lines().collect::<Result<HashSet<_>, _>>()?);
    }

    let res = summary_sa_info(cli.bam, cli.threads, names)?;

    let num_sa = res
        .par_iter()
        .filter(|(_, values)| values.len() > 1)
        .count();

    log::info!("Total number of reads: {}", res.len());
    log::info!("Number of reads with SA tag: {}", num_sa);
    log::info!("Number of reads without SA tag: {}", res.len() - num_sa);

    let file = File::create("sa_summary.json")?;
    let mut writer = std::io::BufWriter::new(file);
    writer.write_all(serde_json::to_string_pretty(&res)?.as_bytes())?;

    log::info!("Summary saved to sa_summary.json");

    let elapsed = start.elapsed();
    log::info!("elapsed time: {:.2?}", elapsed);
    Ok(())
}
