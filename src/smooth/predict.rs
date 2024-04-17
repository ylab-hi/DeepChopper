use std::ops::Range;
use std::path::PathBuf;

use crate::smooth::{ascii_list2str, id_list2seq_i64};
use crate::utils::{get_label_region, summary_predict_i64};

use super::majority_voting;
use anyhow::Result;
use candle_core::{self, pickle};
use log::info;
use pyo3::prelude::*;
use rayon::prelude::*;
use std::collections::HashMap;
use walkdir::WalkDir;

#[pyclass]
#[derive(Debug, Default)]
pub struct Predict {
    prediction: Vec<i8>,
    seq: String,
    id: String,
    is_truncated: bool,
    qual: Option<String>,
}

#[pymethods]
impl Predict {
    #[new]
    pub fn new(
        prediction: Vec<i8>,
        seq: String,
        id: String,
        is_truncated: bool,
        qual: Option<String>,
    ) -> Self {
        Self {
            prediction,
            seq,
            id,
            is_truncated,
            qual,
        }
    }

    #[pyo3(name = "prediction_region")]
    pub fn py_prediction_region(&self) -> Vec<(usize, usize)> {
        get_label_region(&self.prediction)
            .par_iter()
            .map(|r| (r.start, r.end))
            .collect()
    }

    #[pyo3(name = "smooth_prediction")]
    pub fn py_smooth_prediction(&self, window_size: usize) -> Vec<(usize, usize)> {
        get_label_region(&majority_voting(&self.prediction, window_size))
            .par_iter()
            .map(|r| (r.start, r.end))
            .collect()
    }

    #[pyo3(name = "smooth_label")]
    pub fn py_smooth_label(&self, window_size: usize) -> Vec<i8> {
        majority_voting(&self.prediction, window_size)
    }

    #[pyo3(name = "smooth_and_slect_intervals")]
    pub fn py_smooth_and_slect_intervals(
        &self,
        smooth_window_size: usize,
        min_interval_size: usize,
        append_interval_number: usize,
    ) -> Vec<(usize, usize)> {
        self.smooth_and_slect_intervals(
            smooth_window_size,
            min_interval_size,
            append_interval_number,
        )
        .par_iter()
        .map(|r| (r.start, r.end))
        .collect()
    }
}

impl Predict {
    pub fn prediction_region(&self) -> Vec<Range<usize>> {
        get_label_region(&self.prediction)
    }

    pub fn smooth_prediction(&self, window_size: usize) -> Vec<Range<usize>> {
        get_label_region(&majority_voting(&self.prediction, window_size))
    }

    pub fn smooth_label(&self, window_size: usize) -> Vec<i8> {
        majority_voting(&self.prediction, window_size)
    }

    pub fn smooth_and_slect_intervals(
        &self,
        smooth_window_size: usize,
        min_interval_size: usize,
        append_interval_number: usize,
    ) -> Vec<Range<usize>> {
        let chop_interals = self.smooth_prediction(smooth_window_size);

        let results = chop_interals
            .par_iter()
            .filter_map(|interval| {
                if interval.end - interval.start >= min_interval_size {
                    return Some(interval.clone());
                }
                None
            })
            .collect::<Vec<_>>();

        if results.len() > append_interval_number {
            return vec![];
        }

        results
    }
}

#[pyfunction]
pub fn load_predicts_from_batch_pts(pt_path: PathBuf, ignore_label: i64) -> Result<Vec<Predict>> {
    // iter over the pt files under the path
    // make sure there is only one pt file
    let pt_files: Vec<_> = WalkDir::new(pt_path)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().map_or(false, |ext| ext == "pt"))
        .collect();

    info!("Found {} pt files", pt_files.len());

    // Use Rayon to process files in parallel
    let result: Result<Vec<_>, _> = pt_files
        .into_par_iter()
        .filter_map(|entry| {
            let path = entry.path();
            match load_predicts_from_batch_pt(path.to_path_buf(), ignore_label) {
                Ok(predicts) => Some(Ok(predicts)),
                Err(e) => {
                    println!(
                        "load pt {} fail caused by Error: {:?}",
                        path.to_string_lossy(),
                        e
                    );
                    None
                }
            }
        })
        .collect();

    result.map(|vectors| vectors.into_iter().flatten().collect())
}

#[pyfunction]
pub fn load_predicts_from_batch_pt(pt_path: PathBuf, ignore_label: i64) -> Result<Vec<Predict>> {
    let tensors = pickle::read_all(pt_path).unwrap();
    let mut tensors_map = HashMap::new();

    for (key, value) in tensors {
        tensors_map.insert(key, value);
    }

    let _predictions = tensors_map.get("prediction").unwrap().argmax(2)?; // shape batch, seq_len
    let predictions = _predictions.to_dtype(candle_core::DType::I64)?;

    let targets = tensors_map.get("target").unwrap(); // shape batch, seq_len
    let seq = tensors_map.get("seq").unwrap();
    let id = tensors_map.get("id").unwrap();

    let predictions_vec = predictions.to_vec2::<i64>()?;
    let targets_vec = targets.to_vec2::<i64>()?;
    let seq_vec = seq.to_vec2::<i64>()?;
    let id_vec = id.to_vec2::<i64>()?;

    let (true_predictions, _true_label) =
        summary_predict_i64(&predictions_vec, &targets_vec, ignore_label);
    let (true_seqs, _) = summary_predict_i64(&seq_vec, &targets_vec, ignore_label);

    let batch_size = true_predictions.len();

    Ok((0..batch_size)
        .into_par_iter()
        .map(|i| {
            let id_data_end = id_vec[i][0] as usize + 2;
            let id_data = &id_vec[i][2..id_data_end];

            let id = ascii_list2str(id_data);
            let is_truncated = id_vec[i][1] != 0;
            let seq = id_list2seq_i64(&true_seqs[i]);
            let prediction = true_predictions[i].par_iter().map(|&x| x as i8).collect();

            let qual = None;
            Predict {
                prediction,
                seq,
                id,
                is_truncated,
                qual,
            }
        })
        .collect::<Vec<_>>())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_load_predict() {
        let data_path = PathBuf::from("./tests/data/eval/chunk0/0.pt");
        let _predicts = load_predicts_from_batch_pt(data_path, -100).unwrap();
        assert_eq!(_predicts.len(), 12);
    }
}
