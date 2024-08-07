use anyhow::Result;
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use pyo3::types::PyType;
use rayon::prelude::*;
use std::io::BufRead;
use std::ops::Deref; // Import the Deref trait

use super::Predict;

use ahash::HashMap;

use crate::default;
use serde::{Deserialize, Serialize};

const FLANK_SIZE_COUNT_PLOYA: usize = 5;

#[pyclass]
#[derive(Debug, Default, Deserialize, Serialize, Clone)]
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
    pub original_intervals: HashMap<String, Vec<(usize, usize)>>,
    #[pyo3(get, set)]
    pub total_truncated: usize,
    #[pyo3(get, set)]
    pub smooth_only_one: Vec<String>,
    #[pyo3(get, set)]
    pub smooth_only_one_with_ploya: Vec<String>,
    #[pyo3(get, set)]
    pub total_predicts: usize,
    #[pyo3(get, set)]
    pub smooth_intervals_relative_pos: Vec<f32>,
}

#[pymethods]
impl StatResult {
    #[allow(clippy::too_many_arguments)]
    #[new]
    pub fn new(
        predicts_with_chop: Vec<String>,
        smooth_predicts_with_chop: Vec<String>,
        smooth_internal_predicts: Vec<String>,
        smooth_intervals: HashMap<String, Vec<(usize, usize)>>,
        original_intervals: HashMap<String, Vec<(usize, usize)>>,
        total_truncated: usize,
        smooth_only_one: Vec<String>,
        smooth_only_one_with_ploya: Vec<String>,
        total_predicts: usize,
        smooth_intervals_relative_pos: Vec<f32>,
    ) -> Self {
        Self {
            predicts_with_chop,
            smooth_predicts_with_chop,
            smooth_internal_predicts,
            smooth_intervals,
            original_intervals,
            total_truncated,
            smooth_only_one,
            smooth_only_one_with_ploya,
            total_predicts,
            smooth_intervals_relative_pos,
        }
    }

    #[classmethod]
    pub fn from_json(_cls: &Bound<'_, PyType>, json_path: String) -> Result<Self> {
        let file = std::fs::File::open(json_path)?;
        let reader = std::io::BufReader::new(file);
        let json_str = reader.lines().map(|line| line.unwrap()).collect::<String>();
        Ok(serde_json::from_str(&json_str)?)
    }

    pub fn selected_predict_by_intervals(&self, interval_number: usize) -> Vec<String> {
        self.smooth_predicts_with_chop
            .par_iter()
            .flat_map(|id| {
                let intervals = self.smooth_intervals.get(id).unwrap();
                if intervals.len() >= interval_number {
                    return Some(id.clone());
                }
                None
            })
            .collect()
    }

    pub fn length_predicts_with_chop(&self) -> Vec<usize> {
        self.predicts_with_chop
            .par_iter()
            .flat_map(|id| {
                self.original_intervals[id]
                    .iter()
                    .map(|r| r.1 - r.0)
                    .collect::<Vec<usize>>()
            })
            .collect()
    }

    pub fn number_predicts_with_chop(&self) -> Vec<usize> {
        self.predicts_with_chop
            .par_iter()
            .map(|id| self.original_intervals[id].len())
            .collect()
    }

    pub fn lenghth_smooth_predicts_with_chop(&self) -> Vec<usize> {
        self.smooth_predicts_with_chop
            .par_iter()
            .flat_map(|id| {
                self.smooth_intervals[id]
                    .iter()
                    .map(|r| r.1 - r.0)
                    .collect::<Vec<usize>>()
            })
            .collect()
    }

    pub fn number_smooth_predicts_with_chop(&self) -> Vec<usize> {
        self.smooth_predicts_with_chop
            .par_iter()
            .map(|id| self.smooth_intervals[id].len())
            .collect()
    }

    #[pyo3(name = "merge")]
    pub fn py_merge(&mut self, other: PyRef<StatResult>) {
        let other: &StatResult = other.deref();
        self.merge(other.clone())
    }

    fn __repr__(&self) -> String {
        format!(
            "StatResult(total_predicts: {},  predicts_with_chop: {}, smooth_predicts_with_chop: {},
                        smooth_internal_predicts: {}, total_truncated: {}, smooth_only_one: {},
                        smooth_ploya_only_one: {})",
            self.total_predicts,
            self.predicts_with_chop.len(),
            self.smooth_predicts_with_chop.len(),
            self.smooth_internal_predicts.len(),
            self.total_truncated,
            self.smooth_only_one.len(),
            self.smooth_only_one_with_ploya.len(),
        )
    }

    fn __getstate__(&self, py: Python) -> PyResult<PyObject> {
        // Serialize the struct to a JSON string
        let serialized = serde_json::to_string(self).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Failed to serialize: {}", e))
        })?;

        // Convert JSON string to Python bytes
        Ok(PyBytes::new_bound(py, serialized.as_bytes()).into())
    }

    fn __setstate__(&mut self, py: Python, state: PyObject) -> PyResult<()> {
        // Expect a bytes object for state
        let state_bytes: &PyBytes = state.extract(py)?;

        // Deserialize the JSON string into the current instance
        *self = serde_json::from_slice(state_bytes.as_bytes()).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Failed to deserialize: {}",
                e
            ))
        })?;

        Ok(())
    }
}

impl StatResult {
    pub fn merge(&mut self, other: StatResult) {
        self.predicts_with_chop.extend(other.predicts_with_chop);
        self.smooth_predicts_with_chop
            .extend(other.smooth_predicts_with_chop);
        self.smooth_internal_predicts
            .extend(other.smooth_internal_predicts);
        self.smooth_intervals.extend(other.smooth_intervals);
        self.original_intervals.extend(other.original_intervals);
        self.total_truncated += other.total_truncated;
        self.smooth_only_one.extend(other.smooth_only_one);
        self.smooth_only_one_with_ploya
            .extend(other.smooth_only_one_with_ploya);
        self.total_predicts += other.total_predicts;
        self.smooth_intervals_relative_pos
            .extend(other.smooth_intervals_relative_pos);
    }
}

#[pyfunction]
pub fn py_collect_statistics_for_predicts_parallel(
    predicts: Vec<PyRef<Predict>>,
    smooth_window_size: usize,
    min_interval_size: usize,
    approved_interval_number: usize,
    internal_threshold: f32,
    ploya_threshold: usize, // 3
) -> Result<StatResult> {
    let predicts = predicts
        .iter()
        .map(|cell| cell.deref())
        .collect::<Vec<&Predict>>();

    collect_statistics_for_predicts(
        &predicts,
        smooth_window_size,
        min_interval_size,
        approved_interval_number,
        internal_threshold,
        ploya_threshold,
    )
}

pub fn collect_statistics_for_predicts<T>(
    predicts: &[T],
    smooth_window_size: usize,
    min_interval_size: usize,
    approved_interval_number: usize,
    internal_threshold: f32,
    ploya_threshold: usize, // 3
) -> Result<StatResult>
where
    T: Deref<Target = Predict> + Sync,
{
    Ok(predicts
        .par_iter()
        .filter_map(|predict| {
            if predict.seq.len() < default::MIN_READ_LEN {
                return None;
            }
            Some(predict)
        })
        .map(|predict| {
            let mut result = StatResult::default();
            result.total_predicts += 1;

            if predict.is_truncated {
                result.total_truncated += 1;
            }

            let predict_regions = predict
                .prediction_region()
                .par_iter()
                .map(|r| (r.start, r.end))
                .collect::<Vec<(usize, usize)>>();

            if !predict_regions.is_empty() {
                result.predicts_with_chop.push(predict.id.clone());
                result
                    .original_intervals
                    .insert(predict.id.clone(), predict_regions);
            }

            let smooth_regions: Vec<(usize, usize)> = predict
                .smooth_and_select_intervals(
                    smooth_window_size,
                    min_interval_size,
                    approved_interval_number,
                )
                .par_iter()
                .map(|r| (r.start, r.end))
                .collect();

            if !smooth_regions.is_empty() {
                result.smooth_predicts_with_chop.push(predict.id.clone());
                result
                    .smooth_intervals
                    .insert(predict.id.clone(), smooth_regions.clone());

                if smooth_regions.len() == 1 {
                    result.smooth_only_one.push(predict.id.clone());

                    let flank_size = FLANK_SIZE_COUNT_PLOYA;
                    // count first 5 bp of start, if has 3 A
                    let count = predict.seq
                        [smooth_regions[0].0.saturating_sub(flank_size)..smooth_regions[0].0]
                        .chars()
                        .filter(|&c| c == 'A')
                        .count();

                    if count >= ploya_threshold {
                        result.smooth_only_one_with_ploya.push(predict.id.clone());
                    }
                }

                for region in &smooth_regions {
                    let relative_pos = region.1 as f32 / predict.seq_len() as f32;
                    result.smooth_intervals_relative_pos.push(relative_pos);
                    if (relative_pos) < internal_threshold {
                        result.smooth_internal_predicts.push(predict.id.clone());
                    }
                }
            }
            result
        })
        .reduce(StatResult::default, |mut a, b| {
            a.merge(b);
            a
        }))
}
