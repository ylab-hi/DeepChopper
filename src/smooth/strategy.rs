use super::blat::blat;
use super::blat::MIN_SEQ_SIZE;

use super::predict::load_predicts_from_batch_pts;
use super::predict::Predict;

use super::collect_statistics_for_predicts;
use super::StatResult;
use crate::output::read_bam_records_parallel;
use crate::output::BamRecord;

use anyhow::Result;
use derive_builder::Builder;
use log::debug;
use log::info;
use rayon::prelude::*;
use std::collections::HashMap;
use std::path::PathBuf;

#[derive(Builder, Debug, Default, Clone)]
pub struct OverlapOptions {
    pub internal_threshold: f32,
    pub overlap_threshold: f32,
    pub blat_threshold: f32,
    pub min_mapping_quality: usize,
    pub smooth_window_size: usize,
    pub min_interval_size: usize,
    pub append_interval_number: usize,
    pub ploya_threshold: usize, // 3
    pub hg38_2bit: PathBuf,
    pub blat_cli: PathBuf,
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

    let overlap = 0.max(min_end - max_start);
    let divide = length2;

    let ratio = overlap as f32 / divide as f32;
    log::debug!("overlap: {}, divide: {}, ratio: {}", overlap, divide, ratio);
    ratio > overlap_threshold
}

pub fn process_one_interval(
    mut overlap_results: HashMap<String, Vec<String>>,
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
                .or_insert_with(Vec::new)
                .push(predict.id.clone());
        } else {
            overlap_results
                .entry("terminal_chop_nosc".to_string())
                .or_insert_with(Vec::new)
                .push(predict.id.clone());

            if predict_seq.len() < MIN_SEQ_SIZE {
                overlap_results
                    .entry("terminal_chop_nosc_cannot_blat".to_string())
                    .or_insert_with(Vec::new)
                    .push(predict.id.clone());
                return Ok(());
            }

            let blat_result = blat(&predict_seq, &options.blat_cli, &options.hg38_2bit, None);
            if blat_result.is_err() {
                overlap_results
                    .entry("terminal_chop_nosc_blat_fail".to_string())
                    .or_insert_with(Vec::new)
                    .push(predict.id.clone());
                return Ok(());
            }

            let blat_result = blat_result.unwrap();
            if blat_result.is_empty() || blat_result[0].identity < options.blat_threshold {
                overlap_results
                    .entry("terminal_chop_nosc_noblat".to_string())
                    .or_insert_with(Vec::new)
                    .push(predict.id.clone());
            }
        }
    } else {
        // internal adapter
        let mut flag = false;
        if bam_record.left_softclip > 0 {
            if has_overlap(
                (0, bam_record.left_softclip),
                (predict_start, predict_end),
                options.overlap_threshold,
            ) {
                flag = true;
                overlap_results
                    .entry("internal_chop_sc".to_string())
                    .or_insert_with(Vec::new)
                    .push(predict.id.clone());
            }
        }

        if !flag && bam_record.right_softclip > 0 {
            if has_overlap(
                (whole_seq_len - bam_record.right_softclip, whole_seq_len),
                (predict_start, predict_end),
                options.overlap_threshold,
            ) {
                flag = true;
                overlap_results
                    .entry("internal_chop_sc".to_string())
                    .or_insert_with(Vec::new)
                    .push(predict.id.clone());
            }
        }

        if !flag {
            overlap_results
                .entry("internal_chop_nosc".to_string())
                .or_insert_with(Vec::new)
                .push(predict.id.clone());

            if predict_seq.len() < MIN_SEQ_SIZE {
                overlap_results
                    .entry("internal_chop_nosc_cannot_blat".to_string())
                    .or_insert_with(Vec::new)
                    .push(predict.id.clone());
            }

            let blat_result = blat(&predict_seq, &options.blat_cli, &options.hg38_2bit, None);

            if blat_result.is_err() {
                overlap_results
                    .entry("internal_chop_nosc_blat_fail".to_string())
                    .or_insert_with(Vec::new)
                    .push(predict.id.clone());
                return Ok(());
            }

            let blat_result = blat_result.unwrap();
            if blat_result.is_empty() || blat_result[0].identity < options.blat_threshold {
                overlap_results
                    .entry("internal_chop_nosc_noblat".to_string())
                    .or_insert_with(Vec::new)
                    .push(predict.id.clone());
            }
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
    let mut overlap_results = HashMap::new();

    if !bam_record.is_mapped {
        overlap_results
            .entry("unmapped_read".to_string())
            .or_insert_with(Vec::new)
            .push(predict.id.clone());
        return Ok(overlap_results);
    }

    if bam_record.mapping_quality < options.min_mapping_quality {
        overlap_results
            .entry("low_mp_read".to_string())
            .or_insert_with(Vec::new)
            .push(predict.id.clone());
        return Ok(overlap_results);
    }

    let intervals = stats.smooth_intervals.get(&predict.id).unwrap();
    let intervals_number = intervals.len();

    if intervals_number <= 3 {
        for interval in intervals {
            process_one_interval(
                overlap_results,
                interval[0],
                interval[1],
                predict,
                bam_record,
                options,
            )?;
        }
    } else {
        overlap_results
            .entry("no_process".to_string())
            .or_insert_with(Vec::new)
            .push(predict.id.clone());
    }
}

pub fn colect_overlap_results_for_predicts<P: AsRef<Path>>(
    bam_file: P,
    prediction_path: P,
    max_batch_size: Option<usize>,
    options: &OverlapOptions,
) -> Result<HashMap<String, Vec<String>>> {
    let bam_records = read_bam_records_parallel(bam_file)?;

    log::info!(
        "Start to collect overlap results for {} predicts",
        bam_records.len()
    );

    let all_predicts = load_predicts_from_batch_pts(prediction_path, -100, max_batch_size)?;

    log::info!(
        "Start to collect overlap results for {} predicts",
        all_predicts.len()
    );

    // get &[Predict] from HashMap<String, Predict>
    let predicts_value: Vec<&Predict> = all_predicts.values().collect();
    let stats = collect_statistics_for_predicts(
        predicts_value,
        options.smooth_window_size,
        options.min_interval_size,
        options.append_interval_number,
        options.internal_threshold,
        options.ploya_threshold,
    )?;

    // save the stats to a file
    let stats_json = serde_json::to_string(self)?;
    let mut stats_file = File::create("stats.json")?;
    let mut writer = BufWriter::new(stats_file);
    writer.write_all(stats_json.as_bytes())?;
    log::info!("Stats saved to stats.json");

    let overlap_results = all_predicts
        .par_iter()
        .map(|(id, predict)| {
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
                acc.entry(key).or_insert_with(Vec::new).extend(value);
            }
            acc
        });

    merged_results
}
