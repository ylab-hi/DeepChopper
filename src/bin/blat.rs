use anyhow::Result;
use clap::Parser;
use rayon::prelude::*;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};

use ahash::HashMap;
use ahash::HashMapExt;
use ahash::HashSet;

use deepchopper::default;
use deepchopper::smooth::*;
use log::info;
use noodles::fastq::record::Record as FastqRecord;

#[derive(Parser, Debug)]
#[command(version, about = "Blat Adapter Seq and Save base Qual", long_about = None)]
struct Cli {
    /// path to the predicts
    #[arg(long = "pdt", value_name = "predicts", action = clap::ArgAction::Append)]
    predicts: Vec<PathBuf>,

    /// threads number
    #[arg(short, long, default_value = "2")]
    threads: Option<usize>,

    /// max predict batch size
    #[arg(short, long)]
    max_batch_size: Option<usize>,

    /// path to selected read id
    #[arg(long = "sr")]
    selected_reads: Option<PathBuf>,

    ///  path to fastq  file
    #[arg(long = "fq")]
    fastq: Option<PathBuf>,

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

    /// prefix for output files
    #[arg(short, long)]
    prefix: Option<String>,

    /// Turn debugging information on
    #[arg(short, long, action = clap::ArgAction::Count)]
    debug: u8,
}

fn read_selected_reads<P: AsRef<Path>>(file: P) -> HashSet<String> {
    let file = File::open(file.as_ref()).unwrap();
    let reader = BufReader::new(file);
    reader
        .lines()
        .map(|line| {
            let line = line.unwrap();
            line.split_whitespace().next().unwrap().to_string()
        })
        .collect()
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

    let mut selected_quals: Vec<f32> = Vec::new();
    let predict_seqs = if let Some(selected_reads) = cli.selected_reads {
        let selected_reads = read_selected_reads(selected_reads);
        info!("Selected reads number: {}", selected_reads.len());

        let mut selected_fq_records: HashMap<String, FastqRecord> = HashMap::new();
        if let Some(fastq_path) = cli.fastq {
            let records = deepchopper::output::read_noodel_records_from_fq_or_zip_fq(&fastq_path)?;
            selected_fq_records = records
                .into_par_iter()
                .filter_map(|record| {
                    let name = String::from_utf8(record.definition().name().to_vec()).unwrap();
                    if selected_reads.contains(&name) {
                        return Some((name, record));
                    }
                    None
                })
                .collect();
        }

        info!("Selected fq records number: {}", selected_fq_records.len());

        let predict_seqs_qual = selected_reads
            .par_iter()
            .filter_map(|read_id| {
                info!("read_id: {}", read_id);

                let predict = all_predicts.get(read_id).unwrap();
                let smooth_intervals = predict.smooth_and_select_intervals(
                    cli.smooth_window_size,
                    cli.min_interval_size,
                    cli.approved_interval_number,
                );

                if smooth_intervals.len() > cli.max_process_intervals || smooth_intervals.is_empty()
                {
                    return None;
                }

                let qual = selected_fq_records.get(read_id).unwrap().quality_scores();

                let result = smooth_intervals
                    .iter()
                    .map(|interval| {
                        let average_qual = qual[interval.start..interval.end]
                            .iter()
                            .map(|&x| x as f32)
                            .sum::<f32>()
                            / (interval.end - interval.start) as f32;
                        (
                            predict.seq[interval.start..interval.end].to_string(),
                            average_qual,
                        )
                    })
                    .collect::<Vec<_>>();

                Some(result)
            })
            .flatten()
            .collect::<Vec<_>>();

        let predict_seqs = predict_seqs_qual
            .iter()
            .map(|(seq, qual)| {
                selected_quals.push(*qual);
                seq.clone()
            })
            .collect::<Vec<_>>();
        predict_seqs
    } else {
        // get &[Predict] from HashMap<String, Predict>
        let predicts_value: Vec<&Predict> = all_predicts.values().collect();

        let predict_seqs = predicts_value
            .par_iter()
            .filter_map(|predict| {
                let smooth_intervals = predict.smooth_and_select_intervals(
                    cli.smooth_window_size,
                    cli.min_interval_size,
                    cli.approved_interval_number,
                );

                if smooth_intervals.len() > cli.max_process_intervals || smooth_intervals.is_empty()
                {
                    return None;
                }

                let result = smooth_intervals
                    .iter()
                    .map(|interval| predict.seq[interval.start..interval.end].to_string())
                    .collect::<Vec<_>>();
                Some(result)
            })
            .flatten()
            .collect::<Vec<_>>();
        predict_seqs
    };

    log::info!("Collect {} predict seqs", predict_seqs.len());

    // write all seqs to fa
    let file_name = format!(
        "{}all_predicts_seq.fa",
        cli.prefix.clone().unwrap_or_default()
    );
    let fa_file = File::create(&file_name)?;
    let mut fa_writer = BufWriter::new(fa_file);

    for (idx, seq) in predict_seqs.iter().enumerate() {
        fa_writer.write_all(format!(">{}\n", idx).as_bytes())?;
        fa_writer.write_all(seq.as_bytes())?;
        fa_writer.write_all(b"\n")?;
    }

    // make sure the fa_writer is flushed
    fa_writer.flush()?;
    log::info!("Write all seqs to {}", &file_name);

    let hg38_2bit = "/projects/b1171/ylk4626/project/scan_data/hg38.2bit".to_string();
    let blat_cli = "/projects/b1171/ylk4626/project/DeepChopper/tmp/blat".to_string();
    let psl_file = format!("{}blat_res.psl", cli.prefix.clone().unwrap_or_default());
    let psl_alignments = blat_for_seq(file_name, blat_cli, hg38_2bit, psl_file)?;

    log::info!("Collect {} psl alignments", psl_alignments.len());

    let psl_alignments_by_qname: HashMap<String, Vec<PslAlignment>> = psl_alignments
        .into_iter()
        .fold(HashMap::new(), |mut acc, al| {
            let qname = al.qname.clone();
            acc.entry(qname).or_default().push(al);
            acc
        });

    let identities = psl_alignments_by_qname
        .par_iter()
        .map(|(_qname, alignments)| alignments[0].identity)
        .collect::<Vec<_>>();

    // save identities to json file
    let file_name = format!(
        "{}all_predicts_blat_identities.json",
        cli.prefix.clone().unwrap_or_default()
    );
    let json_file = File::create(&file_name)?;
    let mut json_writer = BufWriter::new(json_file);
    json_writer.write_all(serde_json::to_string(&identities)?.as_bytes())?;
    log::info!("Write all identities to {}", &file_name);

    if !selected_quals.is_empty() {
        let file_name = format!(
            "{}predicts_base_quals.json",
            cli.prefix.clone().unwrap_or_default()
        );
        let json_file = File::create(&file_name)?;
        let mut json_writer = BufWriter::new(json_file);
        json_writer.write_all(serde_json::to_string(&selected_quals)?.as_bytes())?;
        log::info!("Write selected reads quals to {}", &file_name);
    }

    let elapsed = start.elapsed();
    log::info!("elapsed time: {:.2?}", elapsed);

    Ok(())
}
