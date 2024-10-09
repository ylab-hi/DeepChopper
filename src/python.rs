use crate::{
    cli::{self, predict_cli},
    default::{BASES, KMER_SIZE, MIN_CHOPED_SEQ_LEN, QUAL_OFFSET, VECTORIZED_TARGET},
    fq_encode::{self, Encoder},
    kmer::{self, vertorize_target},
    output::{self, write_json, write_parquet},
    smooth::{self},
    stat,
    types::{Element, Id2KmerTable, Kmer2IdTable},
    utils,
};
use anyhow::Result;
use bstr::BString;
use needletail::Sequence;
use numpy::{IntoPyArray, PyArray2, PyArray3};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use pyo3::prelude::*;
use rayon::prelude::*;
use std::str::FromStr;
use std::{ops::Range, path::PathBuf};

use ahash::HashMap;
use log::{debug, error, info, warn};

#[pyfunction]
fn encode_qual(qual: String, qual_offset: u8) -> Vec<u8> {
    let quals = qual.as_bytes();
    quals
        .par_iter()
        .map(|&q| {
            // Convert ASCII to Phred score for Phred+33 encoding
            q - qual_offset
        })
        .collect()
}

#[pyfunction]
fn test_log() {
    debug!("debug Hello from Rust!");
    info!("info Hello from Rust!");
    warn!("warn Hello from Rust!");
    error!("error Hello from Rust!");
}

fn register_default_module(parent_module: &Bound<'_, PyModule>) -> PyResult<()> {
    let child_module = PyModule::new_bound(parent_module.py(), "default")?;
    child_module.add("QUAL_OFFSET", QUAL_OFFSET)?;
    child_module.add("BASES", String::from_utf8_lossy(BASES))?;
    child_module.add("KMER_SIZE", KMER_SIZE)?;
    child_module.add("VECTORIZED_TARGET", VECTORIZED_TARGET)?;
    parent_module.add_submodule(&child_module)?;
    Ok(())
}

#[pymethods]
impl fq_encode::TensorEncoder {
    #[new]
    fn py_new(
        option: fq_encode::FqEncoderOption,
        tensor_max_width: Option<usize>,
        tensor_max_seq_len: Option<usize>,
    ) -> Self {
        fq_encode::TensorEncoder::new(option, tensor_max_width, tensor_max_seq_len)
    }
}

#[pymethods]
impl fq_encode::JsonEncoder {
    #[new]
    fn py_new(option: fq_encode::FqEncoderOption) -> Self {
        fq_encode::JsonEncoder::new(option)
    }
}

#[pymethods]
impl fq_encode::ParquetEncoder {
    #[new]
    fn py_new(option: fq_encode::FqEncoderOption) -> Self {
        fq_encode::ParquetEncoder::new(option)
    }
}

#[pyfunction]
fn summary_fx_record_len(path: PathBuf) -> Result<Vec<usize>> {
    stat::summary_fx_record_len(path)
}

#[pyfunction]
fn summary_bam_record_len(path: PathBuf) -> Result<Vec<usize>> {
    stat::summary_bam_record_len(path)
}

#[pyclass(name = "RecordData")]
struct PyRecordData(fq_encode::RecordData);

impl From<fq_encode::RecordData> for PyRecordData {
    fn from(data: fq_encode::RecordData) -> Self {
        Self(data)
    }
}

// Implement FromPyObject for PyRecordData
impl<'py> FromPyObject<'py> for PyRecordData {
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        // Assuming Python objects are tuples of (id, seq, qual)
        let (id, seq, qual): (String, String, String) = ob.extract()?;
        Ok(PyRecordData(fq_encode::RecordData {
            id: id.into(),
            seq: seq.into(),
            qual: qual.into(),
        }))
    }
}

#[pymethods]
impl PyRecordData {
    #[new]
    fn new(id: String, seq: String, qual: String) -> Self {
        Self(fq_encode::RecordData {
            id: id.into(),
            seq: seq.into(),
            qual: qual.into(),
        })
    }

    #[getter]
    fn id(&self) -> String {
        self.0.id.to_string()
    }

    #[setter]
    fn set_id(&mut self, id: String) {
        self.0.id = id.into();
    }

    #[getter]
    fn seq(&self) -> String {
        self.0.seq.to_string()
    }

    #[setter]
    fn set_seq(&mut self, seq: String) {
        self.0.seq = seq.into();
    }

    #[getter]
    fn qual(&self) -> String {
        self.0.qual.to_string()
    }
    #[setter]
    fn set_qual(&mut self, qual: String) {
        self.0.qual = qual.into();
    }
}

#[pyfunction]
fn extract_records_by_ids(ids: Vec<String>, path: PathBuf) -> Result<Vec<PyRecordData>> {
    let ids: Vec<BString> = ids.into_par_iter().map(|id| id.into()).collect();
    output::extract_records_by_ids(&ids, path).map(|records| {
        records
            .into_par_iter()
            .map(|record| record.into())
            .collect()
    })
}

#[pyfunction]
fn write_fq(records_data: Vec<PyRecordData>, file_path: Option<PathBuf>) -> Result<()> {
    let records: Vec<fq_encode::RecordData> = records_data
        .into_par_iter()
        .map(|py_record| py_record.0)
        .collect();
    output::write_fq(&records, file_path)
}

#[pyfunction]
fn write_fq_parallel(
    records_data: Vec<PyRecordData>,
    file_path: PathBuf,
    threads: usize,
) -> Result<()> {
    let records: Vec<fq_encode::RecordData> = records_data
        .into_par_iter()
        .map(|py_record| py_record.0)
        .collect();

    output::write_zip_fq_parallel(&records, file_path, Some(threads))
}

#[pyfunction]
fn kmerids_to_seq(
    kmer_ids: Vec<Element>,
    id2kmer_table: HashMap<Element, String>,
) -> Result<String> {
    // let kmer
    let id2kmer_table: Id2KmerTable = id2kmer_table
        .par_iter()
        .map(|(k, v)| (*k, v.as_bytes().to_vec()))
        .collect();
    kmer::kmerids_to_seq(&kmer_ids, id2kmer_table).map(|x| String::from_utf8_lossy(&x).to_string())
}

#[pyfunction]
fn to_original_targtet_region(start: usize, end: usize, k: usize) -> (usize, usize) {
    let original = kmer::to_original_targtet_region(&(start..end), k);
    (original.start, original.end)
}

#[pyfunction]
fn to_kmer_target_region(
    start: usize,
    end: usize,
    k: usize,
    seq_len: Option<usize>,
) -> Result<(usize, usize)> {
    kmer::to_kmer_target_region(&(start..end), k, seq_len).map(|r| (r.start, r.end))
}

#[pyfunction]
fn seq_to_kmers_and_offset(
    seq: String,
    kmer_size: usize,
    overlap: bool,
) -> Result<Vec<(String, (usize, usize))>> {
    let result = kmer::seq_to_kmers_and_offset(seq.as_bytes(), kmer_size, overlap)?;
    Ok(result
        .par_iter()
        .map(|(s, (start, end))| (String::from_utf8_lossy(s).to_string(), (*start, *end)))
        .collect())
}

#[pyfunction]
fn splite_qual_by_offsets(target: Vec<usize>, offsets: Vec<(usize, usize)>) -> Result<Vec<usize>> {
    kmer::splite_qual_by_offsets(&target, &offsets)
}

#[pyfunction]
fn seq_to_kmers(seq: String, k: usize, overlap: bool) -> Vec<String> {
    let normalized_seq = seq.as_bytes().normalize(false);
    kmer::seq_to_kmers(&normalized_seq, k, overlap)
        .par_iter()
        .map(|s| String::from_utf8_lossy(s).to_string())
        .collect()
}

#[pyfunction]
fn kmers_to_seq(kmers: Vec<String>) -> Result<String> {
    let kmers_as_bytes: Vec<&[u8]> = kmers.par_iter().map(|s| s.as_bytes()).collect();
    Ok(String::from_utf8_lossy(&kmer::kmers_to_seq(kmers_as_bytes)?).to_string())
}

#[pyfunction]
fn generate_kmers_table(base: String, k: usize) -> Kmer2IdTable {
    let base = base.as_bytes();
    kmer::generate_kmers_table(base, k as u8)
}

#[pyfunction]
fn generate_kmers(base: String, k: usize) -> Vec<String> {
    let base = base.as_bytes();
    kmer::generate_kmers(base, k as u8)
        .into_iter()
        .map(|s| String::from_utf8_lossy(&s).to_string())
        .collect()
}

#[pyfunction]
fn normalize_seq(seq: String, iupac: bool) -> String {
    String::from_utf8_lossy(&seq.as_bytes().normalize(iupac)).to_string()
}

#[allow(clippy::too_many_arguments, clippy::type_complexity)]
#[pyfunction]
fn encode_fq_paths_to_tensor(
    py: Python,
    fq_paths: Vec<PathBuf>,
    k: usize,
    bases: String,
    qual_offset: usize,
    vectorized_target: bool,
    parallel_for_files: bool,
    max_width: Option<usize>,
    max_seq_len: Option<usize>,
) -> Result<(
    Bound<'_, PyArray3<Element>>,
    Bound<'_, PyArray3<Element>>,
    Bound<'_, PyArray2<Element>>,
    HashMap<String, Element>,
)> {
    let option = fq_encode::FqEncoderOptionBuilder::default()
        .kmer_size(k as u8)
        .bases(bases.as_bytes().to_vec())
        .qual_offset(qual_offset as u8)
        .vectorized_target(vectorized_target)
        .build()?;

    let mut encoder = fq_encode::TensorEncoderBuilder::default()
        .option(option)
        .tensor_max_width(max_width.unwrap_or(0))
        .tensor_max_seq_len(max_seq_len.unwrap_or(0))
        .build()?;

    let ((input, target), qual) = encoder.encode_multiple(&fq_paths, parallel_for_files)?;

    let kmer2id: HashMap<String, Element> = encoder
        .kmer2id_table
        .par_iter()
        .map(|(k, v)| (String::from_utf8_lossy(k).to_string(), *v))
        .collect();

    Ok((
        input.into_pyarray_bound(py),
        target.into_pyarray_bound(py),
        qual.into_pyarray_bound(py),
        kmer2id,
    ))
}

#[allow(clippy::too_many_arguments, clippy::type_complexity)]
#[pyfunction]
fn encode_fq_path_to_tensor(
    py: Python,
    fq_path: PathBuf,
    k: usize,
    bases: String,
    qual_offset: usize,
    vectorized_target: bool,
    max_width: Option<usize>,
    max_seq_len: Option<usize>,
) -> Result<(
    Bound<'_, PyArray3<Element>>,
    Bound<'_, PyArray3<Element>>,
    Bound<'_, PyArray2<Element>>,
    HashMap<String, Element>,
)> {
    let option = fq_encode::FqEncoderOptionBuilder::default()
        .kmer_size(k as u8)
        .bases(bases.as_bytes().to_vec())
        .qual_offset(qual_offset as u8)
        .vectorized_target(vectorized_target)
        .build()?;

    let mut encoder = fq_encode::TensorEncoderBuilder::default()
        .option(option)
        .tensor_max_width(max_width.unwrap_or(0))
        .tensor_max_seq_len(max_seq_len.unwrap_or(0))
        .build()?;

    let ((input, target), qual) = encoder.encode(fq_path)?;

    let kmer2id: HashMap<String, Element> = encoder
        .kmer2id_table
        .par_iter()
        .map(|(k, v)| (String::from_utf8_lossy(k).to_string(), *v))
        .collect();

    Ok((
        input.into_pyarray_bound(py),
        target.into_pyarray_bound(py),
        qual.into_pyarray_bound(py),
        kmer2id,
    ))
}

#[pyfunction]
fn encode_fq_path_to_json(
    fq_path: PathBuf,
    k: usize,
    bases: String,
    qual_offset: usize,
    vectorized_target: bool,
    result_path: Option<PathBuf>,
) -> Result<()> {
    let option = fq_encode::FqEncoderOptionBuilder::default()
        .kmer_size(k as u8)
        .bases(bases.as_bytes().to_vec())
        .qual_offset(qual_offset as u8)
        .vectorized_target(vectorized_target)
        .build()?;

    let mut encoder = fq_encode::JsonEncoderBuilder::default()
        .option(option)
        .build()?;

    let result = encoder.encode(&fq_path)?;

    // result file is fq_path with .parquet extension
    let json_path = if let Some(path) = result_path {
        if path.with_extension("json").exists() {
            warn!("{} already exists, overwriting", path.display());
        }
        path.with_extension("json")
    } else {
        fq_path.with_extension("json")
    };
    write_json(json_path, result)?;
    Ok(())
}

#[pyfunction]
fn encode_fq_path_to_parquet_chunk(
    fq_path: PathBuf,
    chunk_size: usize,
    parallel: bool,
    bases: String,
    qual_offset: usize,
    vectorized_target: bool,
) -> Result<()> {
    let option = fq_encode::FqEncoderOptionBuilder::default()
        .kmer_size(0)
        .bases(bases.as_bytes().to_vec())
        .qual_offset(qual_offset as u8)
        .vectorized_target(vectorized_target)
        .build()?;

    let mut encoder = fq_encode::ParquetEncoderBuilder::default()
        .option(option)
        .build()?;
    encoder.encode_chunk(&fq_path, chunk_size, parallel)?;
    Ok(())
}

#[pyfunction]
fn encode_fq_path_to_parquet(
    fq_path: PathBuf,
    k: usize,
    bases: String,
    qual_offset: usize,
    vectorized_target: bool,
    result_path: Option<PathBuf>,
) -> Result<()> {
    let option = fq_encode::FqEncoderOptionBuilder::default()
        .kmer_size(0)
        .bases(bases.as_bytes().to_vec())
        .qual_offset(qual_offset as u8)
        .vectorized_target(vectorized_target)
        .build()?;

    let mut encoder = fq_encode::ParquetEncoderBuilder::default()
        .option(option)
        .build()?;
    let (record_batch, schema) = encoder.encode(&fq_path)?;

    // result file is fq_path with .parquet extension
    let parquet_path = if let Some(path) = result_path {
        if path.with_extension("parquet").exists() {
            warn!("{} already exists, overwriting", path.display());
        }
        path.with_extension("parquet")
    } else {
        fq_path.with_extension("parquet")
    };
    write_parquet(parquet_path, record_batch, schema)?;
    Ok(())
}

#[pyfunction]
fn encode_fq_paths_to_parquet(
    fq_path: Vec<PathBuf>,
    k: usize,
    bases: String,
    qual_offset: usize,
    vectorized_target: bool,
) -> Result<()> {
    fq_path.iter().for_each(|path| {
        encode_fq_path_to_parquet(
            path.clone(),
            k,
            bases.clone(),
            qual_offset,
            vectorized_target,
            None,
        )
        .unwrap();
    });
    Ok(())
}

#[pyfunction]
fn summary_predict(
    predictions: Vec<Vec<i8>>,
    labels: Vec<Vec<i8>>,
    ignore_label: i8,
) -> (Vec<Vec<i8>>, Vec<Vec<i8>>) {
    utils::summary_predict(&predictions, &labels, ignore_label)
}

#[allow(clippy::too_many_arguments)]
#[pyfunction]
fn collect_and_split_dataset(
    internal_fq_path: PathBuf,
    terminal_fq_path: PathBuf,
    negative_fq_path: PathBuf,
    total_reads: f32,
    train_ratio: f32, // 0.8
    val_ratio: f32,   // 0.1
    test_ratio: f32,  // 0.1
    iternal_adapter_ratio: f32,
    positive_ratio: f32,
    prefix: Option<&str>,
) -> Result<()> {
    utils::collect_and_split_dataset(
        internal_fq_path,
        terminal_fq_path,
        negative_fq_path,
        total_reads,
        train_ratio,
        val_ratio,
        test_ratio,
        iternal_adapter_ratio,
        positive_ratio,
        prefix,
    )
}

#[allow(clippy::too_many_arguments)]
#[pyfunction]
fn collect_and_split_dataset_with_natural_terminal_adapters(
    internal_fq_path: PathBuf,
    terminal_fq_path: PathBuf,
    natural_terminal_fq_path: PathBuf,
    negative_fq_path: PathBuf,
    total_reads: f32,
    train_ratio: f32,                    // 0.8
    val_ratio: f32,                      // 0.1
    test_ratio: f32,                     // 0.1
    iternal_adapter_ratio: f32,          // 0.5
    natural_terminal_adapter_ratio: f32, // 0.5
    positive_ratio: f32,
    prefix: Option<&str>,
) -> Result<()> {
    utils::collect_and_split_dataset_with_natural_terminal_adapters(
        internal_fq_path,
        terminal_fq_path,
        natural_terminal_fq_path,
        negative_fq_path,
        total_reads,
        train_ratio,
        val_ratio,
        test_ratio,
        iternal_adapter_ratio,
        natural_terminal_adapter_ratio,
        positive_ratio,
        prefix,
    )
}

#[pyfunction]
fn get_label_region(labels: Vec<i8>) -> Vec<(usize, usize)> {
    utils::get_label_region(&labels)
        .par_iter()
        .map(|r| (r.start, r.end))
        .collect()
}

#[pyfunction]
fn smooth_label_region(
    labels: Vec<i8>,
    smooth_window_size: usize,
    min_interval_size: usize,
    approved_interval_number: usize,
) -> Vec<(usize, usize)> {
    utils::smooth_label_region(
        &labels,
        smooth_window_size,
        min_interval_size,
        approved_interval_number,
    )
    .par_iter()
    .map(|r| (r.start, r.end))
    .collect()
}

#[allow(clippy::type_complexity)]
#[pyfunction]
fn remove_intervals_and_keep_left(
    seq: String,
    intervals: Vec<(usize, usize)>,
) -> Result<(Vec<String>, Vec<(usize, usize)>)> {
    let intervals: Vec<Range<usize>> = intervals
        .par_iter()
        .map(|(start, end)| *start..*end)
        .collect();

    let (seqs, intevals) = output::remove_intervals_and_keep_left(seq.as_bytes(), &intervals)?;
    Ok((
        seqs.par_iter().map(|s| s.to_string()).collect(),
        intevals.par_iter().map(|r| (r.start, r.end)).collect(),
    ))
}

#[pyfunction]
fn write_predicts(
    dataset: PathBuf,
    output_fq_path: PathBuf,
    predicts: Vec<Vec<u8>>,
    length_between_intervals_for_merge: usize, // 1
    min_interval_length_threshold: usize,      // 1
    min_interval_length_for_discard: usize,    // 0
) -> Result<()> {
    let file = std::fs::File::open(dataset).unwrap();
    let builder = ParquetRecordBatchReaderBuilder::try_new(file).unwrap();
    let mut reader = builder.build().unwrap();
    let record_batch = reader.next().unwrap().unwrap();

    info!("Read {} records.", record_batch.num_rows());

    let id_column = record_batch.column_by_name("id").unwrap();
    let seq_column = record_batch.column_by_name("seq").unwrap();
    let qual_column = record_batch.column_by_name("qual").unwrap();

    let result = (0..record_batch.num_rows())
        .into_par_iter()
        .map(|i| {
            let predict = &predicts[i].iter().map(|x| *x as i8).collect::<Vec<_>>();
            let smooth_predict = utils::smooth_label_region(
                predict,
                length_between_intervals_for_merge,
                min_interval_length_threshold,
                min_interval_length_for_discard,
            );

            let id = id_column
                .as_any()
                .downcast_ref::<arrow::array::StringArray>()
                .unwrap();
            let seq = seq_column
                .as_any()
                .downcast_ref::<arrow::array::StringArray>()
                .unwrap();
            let qual = qual_column
                .as_any()
                .downcast_ref::<arrow::array::ListArray>()
                .unwrap();

            let current_qual = qual.value(i);
            let qual_array = current_qual
                .as_any()
                .downcast_ref::<arrow::array::Int32Array>()
                .unwrap();
            // Convert the Int32Array for this row into a Vec<i32>
            let qual_len = qual_array.len();
            let qual_vec: Vec<u8> = (0..qual_len)
                .map(|j| qual_array.value(j) as u8 + QUAL_OFFSET)
                .collect();

            let records = output::split_records_by_remove_interval(
                seq.value(i).into(),
                id.value(i).into(),
                &qual_vec,
                &smooth_predict,
                MIN_CHOPED_SEQ_LEN,
                false,
            )
            .unwrap();
            records
        })
        .flatten()
        .collect::<Vec<_>>();

    // output::write_fq(&result, Some("tt.fq".into()))?;
    output::write_zip_fq_parallel(&result, output_fq_path, None)?;
    Ok(())
}

#[pyfunction]
fn convert_multiple_fqs_to_one_fq(
    paths: Vec<PathBuf>,
    result_path: PathBuf,
    parallel: bool,
) -> Result<()> {
    if paths.is_empty() {
        return Ok(());
    }

    let is_zip = paths[0].extension().unwrap() == "gz";

    if is_zip {
        output::convert_multiple_fqs_to_one_zip_fq(&paths, result_path, parallel)?;
    } else {
        output::convert_multiple_zip_fqs_to_one_zip_fq(&paths, result_path, parallel)?;
    }

    Ok(())
}

#[pyfunction]
fn reverse_complement(seq: String) -> String {
    String::from_utf8(seq.as_bytes().reverse_complement()).unwrap()
}

#[pyfunction]
fn id_list2seq(ids: Vec<u8>) -> String {
    smooth::id_list2seq(&ids)
}

#[pyfunction]
fn majority_voting(labels: Vec<i8>, window_size: usize) -> Vec<i8> {
    smooth::majority_voting(&labels, window_size)
}

#[pyfunction]
fn parse_psl_by_qname(file_path: PathBuf) -> Result<HashMap<String, Vec<smooth::PslAlignment>>> {
    smooth::parse_psl_by_qname(file_path)
}

/// CLI func exported to Python
#[pyfunction(name = "predict_cli")]
#[pyo3(signature = (
    predicts,
    fq,
    smooth_window_size=21,
    min_interval_size=13,
    approved_interval_number=20,
    max_process_intervals=4,
    min_read_length_after_chop=20,
    output_chopped_seqs=false,
    chop_type="all",
    threads=2,
    output_prefix=None,
    max_batch_size=None,
))]
fn py_predict_cli(
    predicts: Vec<PathBuf>,
    fq: PathBuf,
    smooth_window_size: usize,
    min_interval_size: usize,
    approved_interval_number: usize,
    max_process_intervals: usize,
    min_read_length_after_chop: usize,
    output_chopped_seqs: bool,
    chop_type: &str,
    threads: Option<usize>,
    output_prefix: Option<String>,
    max_batch_size: Option<usize>,
) -> Result<()> {
    let chop_type = output::ChopType::from_str(chop_type).unwrap();

    let options = cli::PredictOptions {
        predicts,
        fq,
        threads,
        max_batch_size,
        smooth_window_size,
        min_interval_size,
        approved_interval_number,
        max_process_intervals,
        min_read_length_after_chop,
        output_chopped_seqs,
        chop_type,
        output_prefix,
    };

    predict_cli(&options)?;
    Ok(())
}

/// A Python module implemented in Rust.
#[pymodule]
fn deepchopper(m: &Bound<'_, PyModule>) -> PyResult<()> {
    pyo3_log::init();
    register_default_module(m)?;

    m.add_function(wrap_pyfunction!(splite_qual_by_offsets, m)?)?;
    m.add_function(wrap_pyfunction!(vertorize_target, m)?)?;
    m.add_function(wrap_pyfunction!(normalize_seq, m)?)?;
    m.add_function(wrap_pyfunction!(seq_to_kmers, m)?)?;
    m.add_function(wrap_pyfunction!(kmers_to_seq, m)?)?;
    m.add_function(wrap_pyfunction!(seq_to_kmers_and_offset, m)?)?;
    m.add_function(wrap_pyfunction!(generate_kmers_table, m)?)?;
    m.add_function(wrap_pyfunction!(generate_kmers, m)?)?;
    m.add_function(wrap_pyfunction!(encode_fq_path_to_tensor, m)?)?;
    m.add_function(wrap_pyfunction!(to_kmer_target_region, m)?)?;
    m.add_function(wrap_pyfunction!(to_original_targtet_region, m)?)?;
    m.add_function(wrap_pyfunction!(kmerids_to_seq, m)?)?;
    m.add_function(wrap_pyfunction!(write_fq, m)?)?;
    m.add_function(wrap_pyfunction!(write_fq_parallel, m)?)?;
    m.add_function(wrap_pyfunction!(encode_fq_paths_to_tensor, m)?)?;
    m.add_function(wrap_pyfunction!(encode_fq_path_to_tensor, m)?)?;
    m.add_function(wrap_pyfunction!(encode_fq_path_to_parquet, m)?)?;
    m.add_function(wrap_pyfunction!(encode_fq_paths_to_parquet, m)?)?;
    m.add_function(wrap_pyfunction!(encode_fq_path_to_parquet_chunk, m)?)?;
    m.add_function(wrap_pyfunction!(encode_fq_path_to_json, m)?)?;
    m.add_function(wrap_pyfunction!(summary_fx_record_len, m)?)?;
    m.add_function(wrap_pyfunction!(summary_bam_record_len, m)?)?;
    m.add_function(wrap_pyfunction!(test_log, m)?)?;
    m.add_function(wrap_pyfunction!(extract_records_by_ids, m)?)?;
    m.add_function(wrap_pyfunction!(encode_qual, m)?)?;

    // add utils
    m.add_function(wrap_pyfunction!(reverse_complement, m)?)?;
    m.add_function(wrap_pyfunction!(summary_predict, m)?)?;
    m.add_function(wrap_pyfunction!(collect_and_split_dataset, m)?)?;
    m.add_function(wrap_pyfunction!(get_label_region, m)?)?;
    m.add_function(wrap_pyfunction!(smooth_label_region, m)?)?;
    m.add_function(wrap_pyfunction!(remove_intervals_and_keep_left, m)?)?;
    m.add_function(wrap_pyfunction!(write_predicts, m)?)?;
    m.add_function(wrap_pyfunction!(convert_multiple_fqs_to_one_fq, m)?)?;
    m.add_function(wrap_pyfunction!(
        collect_and_split_dataset_with_natural_terminal_adapters,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(output::left_right_soft_clip, m)?)?;
    m.add_function(wrap_pyfunction!(output::py_read_bam_records, m)?)?;
    m.add_function(wrap_pyfunction!(output::py_read_bam_records_parallel, m)?)?;

    m.add_function(wrap_pyfunction!(
        smooth::py_collect_statistics_for_predicts_parallel,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(smooth::test_predicts, m)?)?;
    m.add_function(wrap_pyfunction!(id_list2seq, m)?)?;
    m.add_function(wrap_pyfunction!(majority_voting, m)?)?;
    m.add_function(wrap_pyfunction!(smooth::load_predicts_from_batch_pt, m)?)?;
    m.add_function(wrap_pyfunction!(smooth::load_predicts_from_batch_pts, m)?)?;
    m.add_function(wrap_pyfunction!(parse_psl_by_qname, m)?)?;

    // add clis
    m.add_function(wrap_pyfunction!(py_predict_cli, m)?)?;

    m.add_class::<PyRecordData>()?;
    m.add_class::<fq_encode::FqEncoderOption>()?;
    m.add_class::<fq_encode::TensorEncoder>()?;
    m.add_class::<fq_encode::JsonEncoder>()?;
    m.add_class::<fq_encode::ParquetEncoder>()?;
    m.add_class::<smooth::Predict>()?;
    m.add_class::<output::BamRecord>()?;
    m.add_class::<smooth::StatResult>()?;
    m.add_class::<smooth::PslAlignment>()?;

    Ok(())
}
