use pyo3::prelude::*;

#[pyfunction]
fn test_string() -> PyResult<String> {
    Ok("Hello from Rust!".to_string())
}

/// A Python module implemented in Rust.
#[pymodule]
fn deepchopper(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(test_string, m)?)?;
    Ok(())
}
