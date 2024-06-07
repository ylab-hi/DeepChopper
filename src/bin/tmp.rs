use anyhow::Result;
use clap::Parser;
use rayon::prelude::*;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::PathBuf;

use deepchopper::smooth::*;

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Cli {
    /// path to the predicts
    psl: PathBuf,

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

    let psls = parse_psl_by_qname(cli.psl).unwrap();

    // only pick top one alignment for each query
    let result: Vec<f32> = psls
        .into_par_iter()
        .map(|(_qname, alignments)| {
            let best_alignment = &alignments[0];

            if best_alignment.identity > 0.6 {
                println!("{}: {}", best_alignment.qname, best_alignment.identity);
            }
            best_alignment.identity
        })
        .collect();

    // save identities to json file
    let file_name = "all_predicts_blat_top1_identities.json";
    let json_file = File::create(file_name)?;
    let mut json_writer = BufWriter::new(json_file);
    json_writer.write_all(serde_json::to_string(&result)?.as_bytes())?;
    log::info!("Write all identities to {}", &file_name);

    let elapsed = start.elapsed();
    log::info!("elapsed time: {:.2?}", elapsed);
    Ok(())
}
