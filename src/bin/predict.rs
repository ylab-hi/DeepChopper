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
use sysinfo::System;

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

/// Get current process memory usage in bytes
///
/// Returns: **Current RSS (Resident Set Size)** - actual physical RAM used NOW
///
/// This is NOT:
/// - MaxRSS (peak RSS) - we calculate that separately by tracking max(current_rss)
/// - VmSize (virtual memory) - we don't track this, it includes unused address space
///
/// What RSS includes:
/// - Actual physical RAM pages mapped to this process
/// - All heap allocations across all threads
/// - Thread stacks for main thread + all Rayon workers
/// - Shared library code pages currently in memory
///
/// Important: Only tracks THIS process (same PID), not other jobs on HPC node.
fn get_memory_usage_bytes(sys: &mut System) -> u64 {
    let pid = sysinfo::Pid::from_u32(std::process::id());

    // Refresh system info to get latest memory stats
    sys.refresh_all();

    if let Some(process) = sys.process(pid) {
        // Returns current RSS (Resident Set Size) in bytes - actual physical RAM used
        process.memory()
    } else {
        0
    }
}

/// Format bytes into human-readable memory size (MB, GB, etc.)
fn format_memory(bytes: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = KB * 1024;
    const GB: u64 = MB * 1024;
    const TB: u64 = GB * 1024;

    if bytes >= TB {
        format!("{:.2} TB", bytes as f64 / TB as f64)
    } else if bytes >= GB {
        format!("{:.2} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.1} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.1} KB", bytes as f64 / KB as f64)
    } else {
        format!("{} B", bytes)
    }
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
                log::debug!("truncate seq prediction, do not process: {}", id);
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

    // Initialize system info for memory monitoring (only for this process, not all processes)
    let mut sys = System::new();
    let initial_mem = get_memory_usage_bytes(&mut sys);
    log::info!("Initial memory usage: {}", format_memory(initial_mem));

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
    let mem_after_predictions = get_memory_usage_bytes(&mut sys);
    log::info!(
        "Collected {} predictions (memory: {}, +{})",
        all_predicts_number,
        format_memory(mem_after_predictions),
        format_memory(mem_after_predictions.saturating_sub(initial_mem))
    );

    log::info!(
        "Processing FASTQ records in chunks of {} (streaming mode to reduce memory)",
        cli.chunk_size
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
    let mut peak_memory = mem_after_predictions;

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

            // Track peak memory and progress logging every 10 chunks (or in debug mode every chunk)
            if cli.debug > 0 || (total_fq_count / cli.chunk_size) % 10 == 0 {
                let current_mem = get_memory_usage_bytes(&mut sys);
                peak_memory = peak_memory.max(current_mem);
                let chunks_processed = total_fq_count / cli.chunk_size;
                log::info!(
                    "Progress: {} chunks ({} reads) -> {} output records | RSS: {}",
                    chunks_processed,
                    total_fq_count,
                    total_output_count,
                    format_memory(current_mem)
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

    // Final memory and timing report
    let final_mem = get_memory_usage_bytes(&mut sys);
    peak_memory = peak_memory.max(final_mem);
    let elapsed = start.elapsed();

    // Peak memory = MaxRSS (maximum RSS observed during execution)
    // Final memory = Current RSS at end
    // Both show actual physical RAM used (not virtual memory)
    log::info!(
        "Completed in {:.2?} | Peak RSS: {} | Final RSS: {}",
        elapsed,
        format_memory(peak_memory),
        format_memory(final_mem)
    );

    Ok(())
}
