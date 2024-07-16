use anyhow::Result;
use clap::Parser;
use deepchopper::output;
use noodles::bgzf;
use parquet::data_type::AsBytes;
use rayon::prelude::*;
use std::io::BufRead;
use std::io::BufReader;
use std::io::Write;
use std::path::Path;
use std::path::PathBuf;

use noodles::fastq;
use std::fs::File;

use ahash::HashSet;

#[derive(Debug, Default)]
enum ChopType {
    Terminal,
    #[default]
    Internal,
}

impl ChopType {
    fn to_byte(&self) -> u8 {
        match self {
            ChopType::Terminal => b'T',
            ChopType::Internal => b'I',
        }
    }

    fn is_terminal(&self) -> bool {
        matches!(self, ChopType::Terminal)
    }
}

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Cli {
    /// path to the fq
    #[arg(value_name = "fq")]
    fq: PathBuf,

    /// terminal chop to be selected
    #[arg(long, action = clap::ArgAction::SetTrue)]
    terminal: bool,

    /// internal chop to be selected
    #[arg(long, action = clap::ArgAction::SetTrue)]
    internal: bool,

    /// names of the selected reads
    #[arg(short, long)]
    names: Option<PathBuf>,

    /// only output the name of the selected reads
    #[arg(short, long, action = clap::ArgAction::SetTrue)]
    print_names: bool,

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

fn select_by_name<P: AsRef<Path>>(
    fq: P,
    output_file: P,
    names: &HashSet<String>,
    threads: Option<usize>,
) -> Result<()> {
    let fq_records = output::read_noodel_records_from_fq_or_zip_fq(fq)?;

    let filter_records = fq_records
        .into_par_iter()
        .filter(|record| {
            let id = String::from_utf8(record.definition().name().as_bytes().to_vec()).unwrap();
            if names.contains(&id) {
                return true;
            }
            false
        })
        .collect::<Vec<_>>();

    output::write_fq_parallel_for_noodle_record(
        &filter_records,
        output_file.as_ref().to_path_buf(),
        threads,
    )?;
    Ok(())
}

fn select_by_type<P: AsRef<Path>>(
    fq: P,
    output_file: P,
    selected_type: ChopType,
    threads: Option<usize>,
) -> Result<()> {
    let decoder = bgzf::Reader::new(File::open(fq.as_ref())?);
    let mut reader = fastq::Reader::new(decoder);

    let fq_records = reader
        .records()
        .par_bridge()
        .filter(|record| {
            let record = record.as_ref().unwrap();
            let id = record.definition().name().as_bytes();
            if id.contains(&b'|') {
                // name|region|anno
                // get anno
                let anno = id.split(|&x| x == b'|').last().unwrap();
                if anno[0] == selected_type.to_byte() {
                    return true;
                }
            }
            false
        })
        .map(|record| record.unwrap())
        .collect::<Vec<_>>();
    output::write_fq_parallel_for_noodle_record(
        &fq_records,
        output_file.as_ref().to_path_buf(),
        threads,
    )?;
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
    let mut selected_type = ChopType::default();

    if cli.terminal && cli.internal {
        log::error!("Cannot select both terminal and internal chop");
        std::process::exit(1);
    } else if cli.terminal {
        log::info!("Selecting terminal chop");
        selected_type = ChopType::Terminal;
    } else if cli.internal {
        log::info!("Selecting internal chop");
        selected_type = ChopType::Internal;
    } else if let Some(names) = cli.names {
        log::info!("Selecting by names");
        let name_file = File::open(names)?;
        let reader = std::io::BufReader::new(name_file);
        let names = reader.lines().collect::<Result<HashSet<String>, _>>()?;

        let output_file = if let Some(prefix) = cli.output_prefix {
            format!("{}.fq.gz", prefix)
        } else {
            "selected.fq.gz".to_string()
        };

        select_by_name(cli.fq, output_file.into(), &names, cli.threads)?;

        let elapsed = start.elapsed();
        log::info!("elapsed time: {:.2?}", elapsed);
        return Ok(());
    }

    if cli.print_names {
        log::info!("Printing names of selected reads");
        let fq_records = output::read_noodel_records_from_fq_or_zip_fq(cli.fq)?;

        let stdout = std::io::stdout();
        let mut writer = std::io::BufWriter::new(stdout.lock());

        let res: HashSet<_> = fq_records
            .into_par_iter()
            .filter_map(|record| {
                let id = record.definition().name().as_bytes();
                if id.contains(&b'|') {
                    // name|region|anno
                    let mut splits = id.split(|&x| x == b'|');
                    let name = splits.next().unwrap();
                    let anno = splits.last().unwrap();
                    if anno[0] == selected_type.to_byte() {
                        return Some(name.to_vec());
                    }
                }
                None
            })
            .collect();

        log::info!("{} reads selected", res.len());
        res.iter().for_each(|name| {
            writer.write_all(name).unwrap();
            writer.write_all(b"\n").unwrap();
        });

        let elapsed = start.elapsed();
        log::info!("elapsed time: {:.2?}", elapsed);
        return Ok(());
    }

    let file_stem = cli.fq.file_stem().unwrap().to_str().unwrap();
    let file_type = if selected_type.is_terminal() {
        "terminal"
    } else {
        "internal"
    };

    let output_file = if let Some(prefix) = cli.output_prefix {
        format!("{}.{}.{}.fq.gz", file_stem, prefix, file_type)
    } else {
        format!("{}.{}.fq.gz", file_stem, file_type)
    };
    select_by_type(cli.fq, output_file.into(), selected_type, cli.threads)?;

    let elapsed = start.elapsed();
    log::info!("elapsed time: {:.2?}", elapsed);
    Ok(())
}
