use std::ops::Range;
use std::path::PathBuf;

use ahash::HashMap;
use ahash::HashMapExt;
use anyhow::Result;
use candle_core::{self, pickle};
use log::info;
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use rayon::prelude::*;
use serde::Deserialize;
use serde::Serialize;
use walkdir::WalkDir;

use crate::default;
use crate::smooth::{ascii_list2str, id_list2seq_i64};
use crate::utils::{get_label_region, summary_predict_generic};
use crate::vis;

use super::majority_voting;

#[pyfunction]
pub fn test_predicts(predicts: Vec<PyRef<Predict>>) {
    predicts.iter().for_each(|predict| {
        println!("id: {}", predict.id);
        println!("seq: {}", predict.seq);
        println!("prediction: {:?}", predict.prediction);
        println!("is_truncated: {}", predict.is_truncated);
    });
}

#[pyclass]
#[derive(Debug, Default, FromPyObject, Deserialize, Serialize)]
pub struct Predict {
    #[pyo3(get, set)]
    pub prediction: Vec<i8>,
    #[pyo3(get, set)]
    pub seq: String,
    #[pyo3(get, set)]
    pub id: String,
    #[pyo3(get, set)]
    pub is_truncated: bool,
    #[pyo3(get, set)]
    pub qual: Option<String>,
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

    fn __repr__(&self) -> String {
        format!(
            "Predict(prediction: {:?}, seq: {}, id: {}, is_truncated: {}, qual: {:?})",
            self.prediction, self.seq, self.id, self.is_truncated, self.qual
        )
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

    #[pyo3(name = "smooth_and_select_intervals")]
    pub fn py_smooth_and_slect_intervals(
        &self,
        smooth_window_size: usize,
        min_interval_size: usize,
        append_interval_number: usize,
    ) -> Vec<(usize, usize)> {
        self.smooth_and_select_intervals(
            smooth_window_size,
            min_interval_size,
            append_interval_number,
        )
        .par_iter()
        .map(|r| (r.start, r.end))
        .collect()
    }

    pub fn seq_len(&self) -> usize {
        self.seq.len()
    }

    pub fn qual_array(&self) -> Vec<u8> {
        if let Some(qual) = &self.qual {
            qual.chars()
                .map(|c| c as u8 - default::QUAL_OFFSET)
                .collect()
        } else {
            vec![]
        }
    }

    pub fn show_info(
        &self,
        smooth_interval: Vec<(usize, usize)>,
        text_width: Option<usize>,
    ) -> String {
        let oreg = self.py_prediction_region();
        let oreg_str = vis::highlight_targets(&self.seq, oreg, text_width);
        let sreg_str = vis::highlight_targets(&self.seq, smooth_interval.clone(), text_width);

        let result = format!(
            "id: {}\nprediction: {:?}\nsmooth_intervals: {:?}\n{}\n\n{}",
            self.id,
            self.prediction_region(),
            smooth_interval,
            oreg_str,
            sreg_str
        );
        result
    }

    fn __getstate__(&self, py: Python) -> Result<PyObject> {
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

    pub fn smooth_and_select_intervals(
        &self,
        smooth_window_size: usize,
        min_interval_size: usize,
        approved_interval_number: usize,
    ) -> Vec<Range<usize>> {
        let chop_intervals = self.smooth_prediction(smooth_window_size);

        let results = chop_intervals
            .par_iter()
            .filter_map(|interval| {
                if interval.end - interval.start >= min_interval_size {
                    return Some(interval.clone());
                }
                None
            })
            .collect::<Vec<_>>();

        if results.len() > approved_interval_number {
            return vec![];
        }

        results
    }
}

#[pyfunction]
pub fn load_predicts_from_batch_pts(
    pt_path: PathBuf,
    ignore_label: i64,
    max_predicts: Option<usize>,
) -> Result<HashMap<String, Predict>> {
    // iter over the pt files under the path
    // makes sure there is only one pt file
    let mut pt_files: Vec<_> = WalkDir::new(&pt_path)
        .into_iter()
        .par_bridge()
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().map_or(false, |ext| ext == "pt"))
        .collect();

    info!(
        "Found {} pt files from {}",
        pt_files.len(),
        pt_path.display()
    );

    if let Some(max_predicts) = max_predicts {
        if pt_files.len() > max_predicts {
            info!("only load first {} pt files", max_predicts);
            pt_files.truncate(max_predicts);
        }
    }

    // Use Rayon to process files in parallel
    let result: Result<Vec<_>> = pt_files
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

    result.map(|vectors| vectors.into_par_iter().flatten().collect())
}

#[pyfunction]
pub fn load_predicts_from_batch_pt(
    pt_path: PathBuf,
    ignore_label: i64,
) -> Result<HashMap<String, Predict>> {
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
        summary_predict_generic(&predictions_vec, &targets_vec, ignore_label);
    let (true_seqs, _) = summary_predict_generic(&seq_vec, &targets_vec, ignore_label);

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
            (
                id.clone(),
                Predict {
                    prediction,
                    seq,
                    id,
                    is_truncated,
                    qual,
                },
            )
        })
        .collect::<HashMap<_, _>>())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_load_predict() {
        let data_path = PathBuf::from("./tests/data/eval/chunk0/0.pt");
        let _predicts = load_predicts_from_batch_pt(data_path, -100).unwrap();
        assert_eq!(_predicts.len(), 12);
        let s = _predicts.values().next().unwrap().seq_len();
        println!("seq_len: {}", s);
    }
}
