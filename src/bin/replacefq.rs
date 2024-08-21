use anyhow::Result;
use clap::Parser;
use rayon::prelude::*;
use std::path::{Path, PathBuf};

use noodles::fastq::record::Record as FastqRecord;

use ahash::HashMap;
use ahash::HashSet;
use bstr::BString;

use deepchopper::output;

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Cli {
    /// path to the dc fq file
    #[arg(long)]
    dcfq: PathBuf,

    #[arg(long)]
    /// path to the do fq file
    dofq: PathBuf,

    /// threads number
    #[arg(short, long, default_value = "2")]
    threads: Option<usize>,

    /// Turn debugging information on
    #[arg(short, long, action = clap::ArgAction::Count)]
    debug: u8,
}

fn load_internal_records<P: AsRef<Path>>(path: P) -> Result<HashMap<BString, FastqRecord>> {
    let all_records = output::read_noodle_records_from_bzip_fq(path)?;

    let internal_records = all_records
        .par_iter()
        .filter_map(|record| {
            let name = record.definition().name().to_owned();
            if name.contains(&b'|') {
                let ctype = name.split(|&c| c == b'|').last().unwrap();
                if ctype == b"I" {
                    Some((name.into(), record.clone()))
                } else {
                    None
                }
            } else {
                None
            }
        })
        .collect();
    Ok(internal_records)
}

fn replace_internal<P: AsRef<Path>>(dc_path: P, do_path: P, threads: Option<usize>) -> Result<()> {
    let internal_records = load_internal_records(dc_path)?;

    let internal_records_names: HashSet<BString> = internal_records
        .keys()
        .par_bridge()
        .map(|key| {
            assert!(key.contains(&b'|'));
            let name = key.split(|&c| c == b'|').next().unwrap();
            name.into()
        })
        .collect();

    log::info!(
        "load dc internal records: {} from original read {}",
        internal_records.len(),
        internal_records_names.len()
    );

    let do_all_records = output::read_noodel_records_from_fq_or_zip_fq(do_path.as_ref())?;

    log::info!("load do records: {}", do_all_records.len());

    let mut res: Vec<FastqRecord> = do_all_records
        .into_par_iter()
        .filter(|record| {
            let name: BString = record.definition().name().to_vec().into();
            !internal_records_names.contains(&name)
        })
        .collect();

    log::info!("get {} do  records after filtering", res.len());

    res.extend(internal_records.into_values());

    log::info!("write {} records", res.len());

    let output = do_path
        .as_ref()
        .file_name()
        .unwrap() // &OsStr
        .to_owned(); // Convert &OsStr to OsString

    let output = Path::new(&output).with_extension("do_terminal.fq.bgz");

    log::info!("output to {}", output.display());

    output::write_fq_parallel_for_noodle_record(&res, output, threads)?;

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

    log::info!("cli: {:?}", cli);

    replace_internal(cli.dcfq, cli.dofq, cli.threads)?;

    let elapsed = start.elapsed();
    log::info!("elapsed time: {:.2?}", elapsed);
    Ok(())
}
