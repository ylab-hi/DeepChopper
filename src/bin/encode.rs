use clap::Parser;
use deepchopper::fq_encode::Encoder;
use deepchopper::fq_encode::{FqEncoderOptionBuilder, TensorEncoderBuilder};
use std::path::PathBuf;

#[derive(Parser)]
#[command(version, about, long_about = None)]
struct Cli {
    /// Sets a custom config file
    #[arg(value_name = "fq")]
    input: PathBuf,

    #[arg(value_name = "fq2")]
    input2: Option<PathBuf>,

    /// Turn debugging information on
    #[arg(short, long, action = clap::ArgAction::Count)]
    debug: u8,
}

fn main() {
    let cli = Cli::parse();
    env_logger::init();

    // rayon::ThreadPoolBuilder::new()
    //     .num_threads(1)
    //     .build_global()
    //     .unwrap();

    let option = FqEncoderOptionBuilder::default()
        .kmer_size(3)
        .vectorized_target(true)
        // .max_width(20000)
        // .max_seq_len(20000)
        .build()
        .unwrap();

    let mut encoder = TensorEncoderBuilder::default()
        .option(option)
        .build()
        .unwrap();

    let ((input, target), qual) = encoder.encode(cli.input).unwrap();

    println!("input shape: {:?}", input.shape());
    println!("target shape: {:?}", target.shape());
    println!("qual shape: {:?}", qual.shape());
}
