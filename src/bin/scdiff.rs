use anyhow::Result;
use bstr::io::BufReadExt;
use clap::Parser;
use deepchopper::output;
use noodles::bam;
use noodles::bgzf;
use rayon::prelude::*;
use std::io::Write;
use std::path::Path;
use std::path::PathBuf;

use bstr::BString;
use noodles::sam::alignment::record::cigar::op::Op;
use std::{fs::File, num::NonZeroUsize, thread};

use ahash::HashMap;
use ahash::HashSet;
use ahash::HashSetExt;

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Cli {
    /// path to the choped bam
    #[arg(value_name = "cpbam")]
    cpbam: PathBuf,

    /// path to the compared bam
    #[arg(value_name = "bam")]
    bam: PathBuf,

    /// path to selected reads
    #[arg(short, long, value_name = "reads")]
    reads: Option<PathBuf>,

    /// threads number
    #[arg(short, long, default_value = "2")]
    threads: Option<usize>,

    /// prefix for output files
    #[arg(short, long)]
    output_prefix: Option<String>,

    /// threshold to compare alignment end
    #[arg(short, long, default_value = "5")]
    alignment_end_diff: usize,

    /// Turn debugging information on
    #[arg(short, long, action = clap::ArgAction::Count)]
    debug: u8,
}

fn load_selected_reads<P: AsRef<Path>>(path: P) -> Result<HashSet<BString>> {
    let file = File::open(path)?;
    let reader = std::io::BufReader::new(file);

    Ok(reader
        .byte_lines()
        .par_bridge()
        .map(|line| {
            let line = line.unwrap();
            BString::new(line)
        })
        .collect())
}

fn compare_right_softclip_for_terminal_chop<P: AsRef<Path>>(
    cpbam: P,
    bam: P,
    align_end_diff: usize,
    threads: Option<usize>,
    selected_reads: Option<PathBuf>,
) -> Result<HashMap<String, isize>> {
    let selected_reads = match selected_reads {
        Some(path) => load_selected_reads(path)?,
        None => HashSet::new(),
    };

    if !selected_reads.is_empty() {
        log::info!("selected reads: {}", selected_reads.len());
    }

    let worker_count = if let Some(threads) = threads {
        std::num::NonZeroUsize::new(threads)
            .unwrap()
            .min(thread::available_parallelism().unwrap_or(NonZeroUsize::MIN))
    } else {
        thread::available_parallelism().unwrap_or(NonZeroUsize::MIN)
    };

    let cpfile = File::open(cpbam.as_ref())?;
    let bamfile = File::open(bam.as_ref())?;

    let bamdecoder = bgzf::MultithreadedReader::with_worker_count(worker_count, bamfile);
    let mut bamreader = bam::io::Reader::from(bamdecoder);
    let _header = bamreader.read_header()?;

    let bam_records: HashMap<_, bam::Record> = bamreader
        .records()
        .par_bridge()
        .filter_map(|result| {
            let record = result.unwrap();
            let qname = BString::new(record.name().unwrap().to_vec());

            let is_mapped = !record.flags().is_unmapped();
            let is_not_secondary = !record.flags().is_secondary();
            let is_primary = !record.flags().is_supplementary();

            let is_selected = selected_reads.is_empty() || selected_reads.contains(&qname);

            if is_primary && is_mapped && is_not_secondary && is_selected {
                return Some((qname, record));
            }
            None
        })
        .collect();

    let cpdecoder = bgzf::MultithreadedReader::with_worker_count(worker_count, cpfile);
    let mut cpreader = bam::io::Reader::from(cpdecoder);
    let _header = cpreader.read_header()?;

    let res = cpreader
        .records()
        .par_bridge()
        .filter_map(|result| {
            let record = result.unwrap();
            let is_mapped = !record.flags().is_unmapped();
            let is_not_secondary = !record.flags().is_secondary();
            let is_primary = !record.flags().is_supplementary();

            let qname = BString::new(record.name().unwrap().to_vec());
            let qname_without_anno = if qname.contains(&b'|') {
                let mut splits = qname.split(|item| *item == b'|');
                let name = BString::new(splits.next().unwrap().to_vec());
                splits.next();
                name
            } else {
                qname
            };

            let is_selected =
                selected_reads.is_empty() || selected_reads.contains(&qname_without_anno);
            if is_primary && is_mapped && is_not_secondary && is_selected {
                return Some((record, qname_without_anno));
            }
            None
        })
        .filter_map(|(record, real_id)| {
            let cp_alignment_span = record
                .cigar()
                .iter()
                .filter_map(|op| {
                    let op = op.unwrap();
                    if op.kind().consumes_reference() {
                        Some(op.len())
                    } else {
                        None
                    }
                })
                .sum::<usize>();
            let cp_alignment_end =
                record.alignment_start().unwrap().unwrap().get() + cp_alignment_span;

            let cp_ops: Vec<Op> = record
                .cigar()
                .iter()
                .collect::<Result<Vec<_>, _>>()
                .unwrap();
            let cp_is_forward = !record.flags().is_reverse_complemented();
            let (mut cp_left_softclip, mut cp_right_softclip) =
                output::_calc_softclips(&cp_ops).unwrap();
            if !cp_is_forward {
                std::mem::swap(&mut cp_left_softclip, &mut cp_right_softclip);
            }

            let bam_record = match bam_records.get(&real_id) {
                Some(record) => record,
                None => return None,
            };

            let bam_alignment_span = bam_record
                .cigar()
                .iter()
                .filter_map(|op| {
                    let op = op.unwrap();
                    if op.kind().consumes_reference() {
                        Some(op.len())
                    } else {
                        None
                    }
                })
                .sum::<usize>();
            let bam_alignment_end =
                bam_record.alignment_start().unwrap().unwrap().get() + bam_alignment_span;

            log::debug!(
                "{}: cp: {}, bam: {} cp_right_sc: {}",
                real_id,
                cp_alignment_end,
                bam_alignment_end,
                cp_right_softclip
            );

            if (cp_alignment_end as isize - bam_alignment_end as isize).abs()
                > align_end_diff as isize
            {
                return None;
            }

            Some((real_id.to_string(), cp_right_softclip as isize))
        })
        .collect::<HashMap<_, _>>();

    log::info!("{} records compared", res.len());
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

    let res = compare_right_softclip_for_terminal_chop(
        cli.cpbam,
        cli.bam,
        cli.alignment_end_diff,
        cli.threads,
        cli.reads,
    )?;

    let output_file = if let Some(prefix) = cli.output_prefix {
        format!("{}.terminal_right_softclip_diff.json", prefix)
    } else {
        "terminal_right_softclip_diff.json".to_string()
    };

    let output = File::create(&output_file)?;
    let mut writer = std::io::BufWriter::new(output);
    writer.write_all(serde_json::to_string(&res)?.as_bytes())?;

    log::info!("output file: {}", output_file);

    let elapsed = start.elapsed();
    log::info!("elapsed time: {:.2?}", elapsed);
    Ok(())
}
