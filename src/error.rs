use pyo3::PyErr;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum EncodingError {
    #[error("An error occurred: {0}")]
    Generic(String),
    #[error("Another error occurred")]
    Another,
    #[error("The sequence is shorter than the k-mer size")]
    SeqShorterThanKmer,
}

impl From<EncodingError> for PyErr {
    fn from(error: EncodingError) -> PyErr {
        use EncodingError::*;
        match error {
            Generic(message) => pyo3::exceptions::PyException::new_err(message),
            Another => pyo3::exceptions::PyException::new_err("Another error occurred"),
            SeqShorterThanKmer => pyo3::exceptions::PyException::new_err(
                "The sequence is shorter than the k-mer size",
            ),
        }
    }
}
