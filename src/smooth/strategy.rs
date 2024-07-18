use super::blat::blat;
use super::blat::MIN_SEQ_SIZE;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use super::predict::load_predicts_from_batch_pts;
use super::predict::Predict;

use super::collect_statistics_for_predicts;
use super::StatResult;
use crate::default;
use crate::output::read_bam_records_parallel;
use crate::output::BamRecord;

use anyhow::Result;
use derive_builder::Builder;
use rayon::prelude::*;
use std::path::PathBuf;

use ahash::HashMap;
use ahash::HashMapExt;

#[derive(Builder, Debug, Default, Clone)]
pub struct OverlapOptions {
    pub internal_threshold: f32,
    pub overlap_threshold: f32,
    pub blat_threshold: f32,
    pub min_mapping_quality: usize,
    pub smooth_window_size: usize,
    pub min_interval_size: usize,
    pub approved_interval_number: usize,
    pub max_process_intervals: usize,
    pub ploya_threshold: usize, // 3
    pub hg38_2bit: PathBuf,
    pub blat_cli: PathBuf,
    pub threads: usize,
}

pub fn has_overlap(
    interval1: (usize, usize),
    interval2: (usize, usize),
    overlap_threshold: f32,
) -> bool {
    let (start1, end1) = interval1;
    let (start2, end2) = interval2;
    let _length1 = end1 - start1;
    let length2 = end2 - start2;

    let max_start = start1.max(start2);
    let min_end = end1.min(end2);

    let _min_start = start1.min(start2);
    let _max_end = end1.max(end2);

    let _cmp: isize = min_end as isize - max_start as isize;
    let overlap = 0.max(_cmp) as usize;

    let divide = length2;

    let ratio = overlap as f32 / divide as f32;

    log::debug!(
        "softclip: {:?}, predict: {:?}, ratio: {}",
        interval1,
        interval2,
        ratio
    );
    ratio > overlap_threshold
}

pub fn process_no_interval(
    overlap_results: &mut HashMap<String, Vec<String>>,
    bam_record: &BamRecord,
    options: &OverlapOptions,
) -> Result<()> {
    if bam_record.left_softclip > options.min_interval_size
        || bam_record.right_softclip > options.min_interval_size
    {
        overlap_results
            .entry("sc_without_chop".to_string())
            .or_default()
            .push(bam_record.qname.clone());
    }

    Ok(())
}

pub fn process_one_interval(
    overlap_results: &mut HashMap<String, Vec<String>>,
    predict_start: usize,
    predict_end: usize,
    predict: &Predict,
    bam_record: &BamRecord,
    options: &OverlapOptions,
) -> Result<()> {
    let predict_seq = &predict.seq[predict_start..predict_end];
    let whole_seq_len = predict.seq.len();

    if (predict_end as f32 / whole_seq_len as f32) > options.internal_threshold {
        // terminal adapter
        if has_overlap(
            (whole_seq_len - bam_record.right_softclip, whole_seq_len),
            (predict_start, predict_end),
            options.overlap_threshold,
        ) {
            overlap_results
                .entry("terminal_chop_sc".to_string())
                .or_default()
                .push(predict.id.clone());
        } else {
            overlap_results
                .entry("terminal_chop_nosc".to_string())
                .or_default()
                .push(predict.id.clone());

            if predict_seq.len() < MIN_SEQ_SIZE {
                overlap_results
                    .entry("terminal_chop_nosc_cannot_blat".to_string())
                    .or_default()
                    .push(predict.id.clone());
                return Ok(());
            }

            let blat_result: std::result::Result<Vec<super::PslAlignment>, anyhow::Error> =
                blat(predict_seq, &options.blat_cli, &options.hg38_2bit, None);
            if blat_result.is_err() {
                overlap_results
                    .entry("terminal_chop_nosc_blat_fail".to_string())
                    .or_default()
                    .push(predict.id.clone());
                return Ok(());
            }

            let blat_result = blat_result.unwrap();
            if blat_result.is_empty() || blat_result[0].identity < options.blat_threshold {
                overlap_results
                    .entry("terminal_chop_nosc_noblat".to_string())
                    .or_default()
                    .push(predict.id.clone());
            }
        }
    } else {
        // internal adapter
        if bam_record.left_softclip > 0
            && has_overlap(
                (0, bam_record.left_softclip),
                (predict_start, predict_end),
                options.overlap_threshold,
            )
        {
            overlap_results
                .entry("internal_chop_sc".to_string())
                .or_default()
                .push(predict.id.clone());

            if bam_record.sa_tag.is_some() {
                overlap_results
                    .entry("internal_chop_sc_sa".to_string())
                    .or_default()
                    .push(predict.id.clone());
            }
            return Ok(());
        }

        if bam_record.right_softclip > 0
            && has_overlap(
                (whole_seq_len - bam_record.right_softclip, whole_seq_len),
                (predict_start, predict_end),
                options.overlap_threshold,
            )
        {
            overlap_results
                .entry("internal_chop_sc".to_string())
                .or_default()
                .push(predict.id.clone());

            if bam_record.sa_tag.is_some() {
                overlap_results
                    .entry("internal_chop_sc_sa".to_string())
                    .or_default()
                    .push(predict.id.clone());
            }
            return Ok(());
        }

        overlap_results
            .entry("internal_chop_nosc".to_string())
            .or_default()
            .push(predict.id.clone());

        if predict_seq.len() < MIN_SEQ_SIZE {
            overlap_results
                .entry("internal_chop_nosc_cannot_blat".to_string())
                .or_default()
                .push(predict.id.clone());
            return Ok(());
        }

        let blat_result = blat(predict_seq, &options.blat_cli, &options.hg38_2bit, None);
        if blat_result.is_err() {
            overlap_results
                .entry("internal_chop_nosc_blat_fail".to_string())
                .or_default()
                .push(predict.id.clone());
            return Ok(());
        }

        let blat_result = blat_result.unwrap();
        if blat_result.is_empty() || blat_result[0].identity < options.blat_threshold {
            overlap_results
                .entry("internal_chop_nosc_noblat".to_string())
                .or_default()
                .push(predict.id.clone());
        }
    }
    Ok(())
}

pub fn collect_overlap_results_for_predict(
    stats: &StatResult,
    predict: &Predict,
    bam_record: &BamRecord,
    options: &OverlapOptions,
) -> Result<HashMap<String, Vec<String>>> {
    let mut overlap_results: HashMap<String, Vec<String>> = HashMap::new();

    if !bam_record.is_mapped {
        overlap_results
            .entry("unmapped_read".to_string())
            .or_default()
            .push(predict.id.clone());
        return Ok(overlap_results);
    }

    if bam_record.is_secondary || bam_record.is_supplementary {
        overlap_results
            .entry("secondary_or_supp_read".to_string())
            .or_default()
            .push(predict.id.clone());
        return Ok(overlap_results);
    }

    if bam_record.mapping_quality < options.min_mapping_quality {
        overlap_results
            .entry("low_mp_read".to_string())
            .or_default()
            .push(predict.id.clone());
        return Ok(overlap_results);
    }

    let empty_intervals = Vec::new();
    let intervals = stats
        .smooth_intervals
        .get(&predict.id)
        .unwrap_or(&empty_intervals);

    let intervals_number = intervals.len();

    log::debug!(
        "pid: {},  intervals: {:?} bam ls: {} rs: {} mp: {}",
        predict.id,
        intervals,
        bam_record.left_softclip,
        bam_record.right_softclip,
        bam_record.mapping_quality
    );

    if intervals_number == 0 {
        process_no_interval(&mut overlap_results, bam_record, options)?;
    } else if intervals_number <= options.max_process_intervals {
        for interval in intervals {
            process_one_interval(
                &mut overlap_results,
                interval.0,
                interval.1,
                predict,
                bam_record,
                options,
            )?;
        }
    } else {
        overlap_results
            .entry("no_process".to_string())
            .or_default()
            .push(predict.id.clone());
    }

    Ok(overlap_results)
}

pub fn collect_overlap_results_for_predicts<P: AsRef<Path>>(
    bam_file: P,
    prediction_path: P,
    max_batch_size: Option<usize>,
    options: &OverlapOptions,
) -> Result<HashMap<String, Vec<String>>> {
    let bam_records = read_bam_records_parallel(bam_file, Some(options.threads))?;

    log::info!("Collect {} bam records", bam_records.len());

    let all_predicts = load_predicts_from_batch_pts(
        prediction_path.as_ref().to_path_buf(),
        default::IGNORE_LABEL,
        max_batch_size,
    )?;

    let all_predicts_number = all_predicts.len();

    log::info!("Collect {} predicts", all_predicts_number);

    // get &[Predict] from HashMap<String, Predict>
    let predicts_value: Vec<&Predict> = all_predicts.values().collect();
    let stats = collect_statistics_for_predicts(
        predicts_value.as_slice(),
        options.smooth_window_size,
        options.min_interval_size,
        options.approved_interval_number,
        options.internal_threshold,
        options.ploya_threshold,
    )?;

    // save the stats to a file
    let stats_json = serde_json::to_string(&stats)?;
    let stats_file_name = format!(
        "stats_pd{}_bt{}.json",
        all_predicts_number,
        max_batch_size.unwrap_or(0)
    );
    let stats_file = File::create(&stats_file_name)?;
    let mut writer = BufWriter::new(stats_file);
    writer.write_all(stats_json.as_bytes())?;
    log::info!("Stats saved to {}", stats_file_name);

    let stats_smooth_intervals_number = stats.smooth_predicts_with_chop.len();
    log::info!(
        "Start to collect overlap results for {} predicts",
        stats_smooth_intervals_number
    );

    // stats .smooth_predicts_with_chop // NOTE:  <Yangyang Li>  Note: change here to all predicts
    let overlap_results = all_predicts
        .par_iter()
        .map(|(id, _predcit)| {
            let predict = all_predicts.get(id).unwrap();
            let bam_record = bam_records.get(id).unwrap();
            let overlap_results =
                collect_overlap_results_for_predict(&stats, predict, bam_record, options)?;
            Ok(overlap_results)
        })
        .collect::<Result<Vec<_>>>()?;

    // merge all the results in parallel use reduce
    let merged_results = overlap_results
        .into_par_iter()
        .reduce_with(|mut acc, result| {
            for (key, value) in result {
                acc.entry(key).or_default().extend(value);
            }
            acc
        })
        .unwrap();

    let overlap_json = serde_json::to_string(&merged_results)?;
    let overlap_file_name = format!(
        "overlap_results_spd{}_pd{}.json",
        stats_smooth_intervals_number, all_predicts_number
    );
    let overlap_file = File::create(&overlap_file_name)?;
    let mut writer = BufWriter::new(overlap_file);
    writer.write_all(overlap_json.as_bytes())?;
    log::info!("overlap_results  saved to {}", overlap_file_name);
    Ok(merged_results)
}
