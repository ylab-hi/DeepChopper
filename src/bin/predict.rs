use std::path::PathBuf;

use ahash::HashMap;
use anyhow::Result;
use bstr::BStr;
use clap::Parser;
use rayon::prelude::*;

use deepchopper::default;
use deepchopper::output;
use deepchopper::output::ChopType;
use deepchopper::smooth::*;

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Cli {
    /// path to the predicts
    #[arg(long = "pdt", value_name = "predicts", action = clap::ArgAction::Append)]
    predicts: Vec<PathBuf>,

    /// path to the integrated fq file
    #[arg(long = "fq", value_name = "fq")]
    fq: PathBuf,

    /// threads number
    #[arg(short, long, default_value = "2")]
    threads: Option<usize>,

    /// max predict batch size
    #[arg(short, long)]
    max_batch_size: Option<usize>,

    /// smooth_window_size
    #[arg(short, long, default_value = "21")]
    smooth_window_size: usize,

    /// min_interval_size
    #[arg(long = "mis", default_value = "13")] // 11, 12, 13 are good
    min_interval_size: usize,

    /// approved_interval_number
    #[arg(short, long, default_value = "20")]
    approved_interval_number: usize,

    /// max process intervals
    #[arg(long = "mpi", default_value = "4")]
    max_process_intervals: usize,

    /// min read length after chop
    #[arg(long = "mcr", default_value = "20")]
    min_read_length_after_chop: usize,

    /// output chopped fq file only
    #[arg(long = "ocq", action=clap::ArgAction::SetTrue)]
    output_chopped_seqs: bool,

    /// selected chopped type
    #[arg(long = "ct",  default_value_t = ChopType::All)]
    chop_type: ChopType,

    /// prefix for output files
    #[arg(short, long)]
    output_prefix: Option<String>,

    /// Turn debugging information on
    #[arg(short, long, action = clap::ArgAction::Count)]
    debug: u8,
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

    log::info!("export {:?} chopped type ", cli.chop_type);

    let all_predicts = cli
        .predicts
        .par_iter()
        .map(|path| {
            load_predicts_from_batch_pts(
                path.to_path_buf(),
                default::IGNORE_LABEL,
                cli.max_batch_size,
            )
        })
        .collect::<Result<Vec<_>>>()?
        .into_par_iter()
        .flatten()
        .collect::<HashMap<_, _>>();

    let all_predicts_number = all_predicts.len();
    log::info!("Collect {} predicts", all_predicts_number);

    // load fq file
    let mut _records = output::read_noodel_records_from_fq_or_zip_fq(&cli.fq)?;
    let fq_records = _records
        .into_par_iter()
        .map(|record| {
            let id = String::from_utf8(record.definition().name().to_vec()).unwrap();
            (id, record)
        })
        .collect::<HashMap<_, _>>();

    log::info!(
        "Load {} fq records from {}",
        fq_records.len(),
        cli.fq.display()
    );

    let res = all_predicts
        .into_par_iter()
        .map(|(id, predict)| {
            let fq_record = fq_records.get(&id).expect("id not found");

            if predict.seq.len() < default::MIN_READ_LEN {
                return Ok(vec![fq_record.clone()]);
            }

            let smooth_intervals = predict.smooth_and_select_intervals(
                cli.smooth_window_size,
                cli.min_interval_size,
                cli.approved_interval_number,
            );

            if smooth_intervals.len() > cli.max_process_intervals || smooth_intervals.is_empty() {
                return Ok(vec![fq_record.clone()]);
            }

            if predict.seq.len() != fq_record.quality_scores().len() {
                // truncate seq prediction, do not process
                return Ok(vec![fq_record.clone()]);
            };

            if cli.output_chopped_seqs {
                output::split_noodle_records_by_intervals(
                    BStr::new(&predict.seq),
                    id.as_bytes().into(),
                    fq_record.quality_scores(),
                    &smooth_intervals,
                )
            } else {
                output::split_noodle_records_by_remove_intervals(
                    BStr::new(&predict.seq),
                    id.as_bytes().into(),
                    fq_record.quality_scores(),
                    &smooth_intervals,
                    cli.min_read_length_after_chop,
                    true, // NOTE: add annotation for terminal or internal chop
                    &cli.chop_type,
                )
            }
        })
        .collect::<Result<Vec<_>>>()?
        .into_par_iter()
        .flatten()
        .collect::<Vec<_>>();

    let output_file = if let Some(prefix) = cli.output_prefix {
        format!(
            "{}.{}pd.{}record.chop.fq.bgz",
            prefix,
            all_predicts_number,
            res.len()
        )
    } else {
        let file_stem = cli.fq.file_stem().unwrap().to_str().unwrap();
        format!(
            "{}.{}pd.{}record.chop.fq.bgz",
            file_stem,
            all_predicts_number,
            res.len()
        )
    };

    if cli.output_chopped_seqs {
        log::info!("Output chopped adapters");
    }

    output::write_fq_parallel_for_noodle_record(&res, output_file.clone().into(), cli.threads)?;

    log::info!("Write {} records to {}", res.len(), output_file);

    let elapsed = start.elapsed();
    log::info!("elapsed time: {:.2?}", elapsed);

    Ok(())
}
