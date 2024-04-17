use anyhow::Result;
use pyo3::prelude::*;
use rayon::prelude::*;

use super::Predict;
use std::collections::HashMap;

#[pyclass]
#[derive(Debug, Default)]
pub struct StatResult {
    #[pyo3(get, set)]
    pub predicts_with_chop: Vec<String>,

    #[pyo3(get, set)]
    pub smooth_predicts_with_chop: Vec<String>,

    #[pyo3(get, set)]
    pub smooth_internal_predicts: Vec<String>,

    #[pyo3(get, set)]
    pub smooth_intervals: HashMap<String, Vec<(usize, usize)>>,

    #[pyo3(get, set)]
    pub total_truncated: usize,
}

#[pymethods]
impl StatResult {
    #[new]
    pub fn new(
        predicts_with_chop: Vec<String>,
        smooth_predicts_with_chop: Vec<String>,
        smooth_internal_predicts: Vec<String>,
        smooth_intervals: HashMap<String, Vec<(usize, usize)>>,
        total_truncated: usize,
    ) -> Self {
        Self {
            predicts_with_chop,
            smooth_predicts_with_chop,
            smooth_internal_predicts,
            smooth_intervals,
            total_truncated,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "StatResult(predicts_with_chop: {}, smooth_predicts_with_chop: {}, total_truncated: {})",
            self.predicts_with_chop.len(),
            self.smooth_predicts_with_chop.len(),
            self.total_truncated
        )
    }
}

#[pyfunction]
pub fn collect_statistics_for_predicts(
    predicts: Vec<PyRef<Predict>>,
    smooth_window_size: usize,
    min_interval_size: usize,
    append_interval_number: usize,
    internal_threshold: f32,
) -> Result<StatResult> {
    let mut predicts_with_chop = Vec::new();
    let mut smooth_predicts_with_chop = Vec::new();
    let mut smooth_intervals = HashMap::new();
    let mut smooth_internal_predicts = Vec::new();
    let mut total_truncated = 0;

    for predict in predicts {
        if predict.is_truncated {
            total_truncated += 1;
        }

        if !predict.prediction_region().is_empty() {
            predicts_with_chop.push(predict.id.clone());
        }

        let smooth_regions: Vec<(usize, usize)> = predict
            .smooth_and_slect_intervals(
                smooth_window_size,
                min_interval_size,
                append_interval_number,
            )
            .par_iter()
            .map(|r| (r.start, r.end))
            .collect();

        if !smooth_regions.is_empty() {
            smooth_predicts_with_chop.push(predict.id.clone());
            for region in &smooth_regions {
                if (region.1 as f32 / predict.seq_len() as f32) < internal_threshold {
                    smooth_internal_predicts.push(predict.id.clone());
                }
            }
            smooth_intervals.insert(predict.id.clone(), smooth_regions);
        }
    }

    Ok(StatResult::new(
        predicts_with_chop,
        smooth_predicts_with_chop,
        smooth_internal_predicts,
        smooth_intervals,
        total_truncated,
    ))
}
