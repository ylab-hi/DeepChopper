use clap::Parser;
use deepchopper::smooth::*;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Cli {
    /// path to the bam file
    #[arg(value_name = "bam")]
    bam: PathBuf,

    /// path to the predicts
    #[arg(value_name = "predicts")]
    predicts: PathBuf,

    /// max predict batch size
    #[arg(short, long)]
    max_batch_size: Option<usize>,

    /// threads number
    #[arg(short, long, default_value = "2")]
    threads: Option<usize>,

    /// internal_threshold
    #[arg(short, long, default_value = "0.9")]
    internal_threshold: f32,

    /// overlap_threshold
    #[arg(short, long, default_value = "0.4")]
    overlap_threshold: f32,

    /// blat_threshold
    #[arg(short, long, default_value = "0.9")]
    blat_threshold: f32,

    /// min_mapping_quality
    #[arg(long = "mmq", default_value = "0")]
    min_mapping_quality: usize,

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

    /// ploya_threshold
    #[arg(short, long, default_value = "3")]
    polya_threshold: usize,

    /// Turn debugging information on
    #[arg(short, long, action = clap::ArgAction::Count)]
    debug: u8,
}

fn main() {
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

    // let bam_file = "/projects/b1171/ylk4626/project/DeepChopper/data/eval/real_data/dorado_without_trim_fqs/VCaP.bam";
    // let prediction_path =
    // "/projects/b1171/ylk4626/project/DeepChopper/logs/eval/runs/vcap/VCaP.fastq_0/predicts/0/";
    let options = OverlapOptionsBuilder::default()
        .internal_threshold(cli.internal_threshold)
        .overlap_threshold(cli.overlap_threshold)
        .blat_threshold(cli.blat_threshold)
        .min_mapping_quality(cli.min_mapping_quality)
        .smooth_window_size(cli.smooth_window_size)
        .min_interval_size(cli.min_interval_size)
        .approved_interval_number(cli.approved_interval_number)
        .max_process_intervals(cli.max_process_intervals)
        .ploya_threshold(cli.polya_threshold)
        .hg38_2bit("/projects/b1171/ylk4626/project/scan_data/hg38.2bit".into())
        .blat_cli("/projects/b1171/ylk4626/project/DeepChopper/tmp/blat".into())
        .threads(cli.threads.unwrap())
        .build()
        .unwrap();

    log::info!("bam: {:?}", cli.bam);
    log::info!("predicts: {:?}", cli.predicts);
    log::info!("max_batch_size: {:?}", cli.max_batch_size);
    log::info!("options: {:?}", options);

    let _overlap_results =
        collect_overlap_results_for_predicts(cli.bam, cli.predicts, cli.max_batch_size, &options)
            .unwrap();

    let elapsed = start.elapsed();
    log::info!("elapsed time: {:.2?}", elapsed);
}
