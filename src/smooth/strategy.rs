use super::blat::MIN_SEQ_SIZE;
use super::predict::Predict;
use crate::output::BamRecord;
use anyhow::Result;
use derive_builder::Builder;
use log::debug;
use log::info;
use std::collections::HashMap;

#[derive(Builder, Debug, Default, Clone)]
pub struct OverlapOptions {
    pub internal_threshold: f32,
    pub overlap_threshold: f32,
    pub blat_threshold: f32,
    pub min_mapping_quality: usize,
}

pub fn check_overlap(
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

    if (predict_end as f32 / whole_seq_len as f32) < options.internal_threshold {
        return Ok(());
    }

    Ok(())
}
