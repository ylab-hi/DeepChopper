use std::fs::File;
use std::num::NonZeroUsize;
use std::path::PathBuf;
use std::sync::Mutex;
use std::thread;

use ahash::HashMap;
use anyhow::Result;
use bstr::BStr;
use clap::Parser;
use noodles::bgzf;
use noodles::fastq;
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

    /// chunk size for streaming processing (reduce for lower memory, increase for better performance)
    /// Typical values: 1000 (low memory), 10000 (balanced), 50000 (high performance)
    #[arg(long = "chunk-size", default_value = "10000")]
    chunk_size: usize,

    /// Turn debugging information on
    #[arg(short, long, action = clap::ArgAction::Count)]
    debug: u8,
}

/// Process a chunk of FASTQ records with their predictions
fn process_chunk(
    chunk: &[noodles::fastq::Record],
    all_predicts: &HashMap<String, Predict>,
    cli: &Cli,
) -> Result<Vec<noodles::fastq::Record>> {
    let results = chunk
        .par_iter()
        .filter_map(|fq_record| {
            let id = String::from_utf8(fq_record.definition().name().to_vec()).unwrap();

            // Look up prediction for this FASTQ record
            let predict = match all_predicts.get(&id) {
                Some(p) => p,
                None => return None, // Skip records without predictions
            };

            if predict.seq.len() < default::MIN_READ_LEN {
                return Some(Ok(vec![fq_record.clone()]));
            }

            let smooth_intervals = predict.smooth_and_select_intervals(
                cli.smooth_window_size,
                cli.min_interval_size,
                cli.approved_interval_number,
            );

            if smooth_intervals.len() > cli.max_process_intervals || smooth_intervals.is_empty() {
                return Some(Ok(vec![fq_record.clone()]));
            }

            if predict.seq.len() != fq_record.quality_scores().len() {
                // truncate seq prediction, do not process
                log::warn!("truncate seq prediction, do not process: {}", id);
                return Some(Ok(vec![fq_record.clone()]));
            }

            let result = if cli.output_chopped_seqs {
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
            };
            Some(result)
        })
        .collect::<Result<Vec<_>>>()?
        .into_par_iter()
        .flatten()
        .collect();

    Ok(results)
}

/// Main function that processes FASTQ records with predictions using streaming mode
/// to minimize memory usage. Records are processed in configurable chunks and written
/// incrementally to reduce memory footprint.
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
    log::info!("Collected {} predictions", all_predicts_number);
    log::info!(
        "Processing FASTQ records in chunks of {} (streaming mode to reduce memory)",
        cli.chunk_size
    );

    // Estimate peak memory usage for user awareness
    let est_chunk_memory_mb = (cli.chunk_size * 500) / 1_048_576; // Rough estimate: ~500 bytes per record
    log::info!(
        "Estimated peak memory per chunk: ~{}MB (vs loading all records at once)",
        est_chunk_memory_mb.max(1)
    );

    // Determine output directory to create temp file in the same filesystem as final output.
    // This prevents "Invalid cross-device link" errors when renaming across filesystems.
    let output_dir = if let Some(ref prefix) = cli.output_prefix {
        std::path::Path::new(prefix)
            .parent()
            .unwrap_or_else(|| std::path::Path::new("."))
            .to_path_buf()
    } else {
        cli.fq
            .parent()
            .unwrap_or_else(|| std::path::Path::new("."))
            .to_path_buf()
    };

    // Create temporary output file in the same directory as the final output
    let temp_output = output_dir.join(format!(".deepchopper_temp_{}.fq.gz", std::process::id()));

    // Set up the writer for incremental writing
    let worker_count = NonZeroUsize::new(cli.threads.unwrap_or(2))
        .map(|count| count.min(thread::available_parallelism().unwrap()))
        .unwrap();
    let sink = File::create(&temp_output)?;
    let encoder = bgzf::io::MultithreadedWriter::with_worker_count(worker_count, sink);
    let writer = Mutex::new(fastq::io::Writer::new(encoder));

    // Stream FASTQ records and process with parallel chunks
    let streaming_reader = output::StreamingFastqReader::new(&cli.fq)?;

    let mut current_chunk = Vec::new();
    let mut total_fq_count = 0;
    let mut total_output_count = 0;

    for record_result in streaming_reader {
        let fq_record = record_result?;
        current_chunk.push(fq_record);
        total_fq_count += 1;

        // Process chunk when it reaches the desired size
        if current_chunk.len() >= cli.chunk_size {
            let chunk_results = process_chunk(&current_chunk, &all_predicts, &cli)?;

            // Write chunk results immediately
            {
                let mut writer_guard = writer.lock().unwrap();
                for record in &chunk_results {
                    writer_guard.write_record(record)?;
                }
                total_output_count += chunk_results.len();
            }

            current_chunk.clear();

            // Progress logging every 10 chunks (or in debug mode every chunk)
            if cli.debug > 0 || (total_fq_count / cli.chunk_size) % 10 == 0 {
                let chunks_processed = total_fq_count / cli.chunk_size;
                log::info!(
                    "Progress: {} chunks ({} reads) processed -> {} output records",
                    chunks_processed,
                    total_fq_count,
                    total_output_count
                );
            }
        }
    }

    // Process remaining records in the last chunk
    if !current_chunk.is_empty() {
        let chunk_results = process_chunk(&current_chunk, &all_predicts, &cli)?;

        // Write remaining results
        {
            let mut writer_guard = writer.lock().unwrap();
            for record in &chunk_results {
                writer_guard.write_record(record)?;
            }
            total_output_count += chunk_results.len();
        }
    }

    // Close the writer by dropping it
    drop(writer);

    log::info!(
        "Processed {} FASTQ records, generated {} output records",
        total_fq_count,
        total_output_count
    );

    // Determine final output filename with correct record count
    let output_file = if let Some(prefix) = cli.output_prefix {
        format!(
            "{}.{}pd.{}record.chop.fq.gz",
            prefix, all_predicts_number, total_output_count
        )
    } else {
        let file_stem = cli.fq.file_stem().unwrap().to_str().unwrap();
        format!(
            "{}.{}pd.{}record.chop.fq.gz",
            file_stem, all_predicts_number, total_output_count
        )
    };

    if cli.output_chopped_seqs {
        log::info!("Output chopped adapters");
    }

    // Rename temp file to final output file
    if let Err(e) = std::fs::rename(&temp_output, &output_file) {
        // Clean up temp file on error
        let _ = std::fs::remove_file(&temp_output);
        return Err(e.into());
    }

    log::info!("Wrote {} records to {}", total_output_count, output_file);

    let elapsed = start.elapsed();
    log::info!("elapsed time: {:.2?}", elapsed);

    Ok(())
}
