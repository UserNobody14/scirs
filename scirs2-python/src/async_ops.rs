//! Async operations for Python
//!
//! This module provides async versions of long-running operations that can be awaited in Python.
//!
//! # Example (Python)
//! ```python
//! import asyncio
//! import scirs2
//! import numpy as np
//!
//! async def main():
//!     # Async FFT for large arrays
//!     data = np.random.randn(1_000_000)
//!     result = await scirs2.fft_async(data)
//!
//!     # Async matrix decomposition
//!     matrix = np.random.randn(1000, 1000)
//!     svd = await scirs2.svd_async(matrix)
//!
//! asyncio.run(main())
//! ```

use pyo3::prelude::*;
use pyo3_asyncio;
use scirs2_numpy::{IntoPyArray, PyArray1, PyArray2, PyArrayMethods};
use crate::error::SciRS2Error;

/// Async FFT operation for large arrays
///
/// This function runs FFT in a background thread and returns a Python awaitable.
/// Useful for large arrays (>100k elements) to avoid blocking the event loop.
#[pyfunction]
pub fn fft_async<'py>(
    py: Python<'py>,
    data: &Bound<'_, PyArray1<f64>>,
) -> PyResult<Bound<'py, PyAny>> {
    let data_vec: Vec<f64> = {
        let binding = data.readonly();
        let arr = binding.as_array();
        arr.to_vec()
    };

    pyo3_asyncio::tokio::future_into_py(py, async move {
        // Run FFT in blocking task
        let result = tokio::task::spawn_blocking(move || {
            // Import FFT implementation
            use scirs2_fft::fft;
            use scirs2_core::Array1;

            let arr = Array1::from_vec(data_vec);
            fft(&arr).map_err(|e| SciRS2Error::ComputationError(format!("FFT failed: {}", e)))
        })
        .await
        .map_err(|e| SciRS2Error::RuntimeError(format!("Task join error: {}", e)))??;

        Python::with_gil(|py| {
            Ok(result.into_pyarray(py).into_any())
        })
    })
}

/// Async SVD operation for large matrices
///
/// This function runs SVD in a background thread and returns a Python awaitable.
/// Useful for large matrices (>500x500) to avoid blocking the event loop.
#[pyfunction]
pub fn svd_async<'py>(
    py: Python<'py>,
    matrix: &Bound<'_, PyArray2<f64>>,
    full_matrices: Option<bool>,
) -> PyResult<Bound<'py, PyAny>> {
    let matrix_vec: Vec<f64> = {
        let binding = matrix.readonly();
        let arr = binding.as_array();
        arr.to_vec()
    };
    let shape = matrix.shape();
    let full_matrices = full_matrices.unwrap_or(true);

    pyo3_asyncio::tokio::future_into_py(py, async move {
        // Run SVD in blocking task
        let result = tokio::task::spawn_blocking(move || {
            use scirs2_linalg::svd_f64_lapack;
            use scirs2_core::Array2;

            let arr = Array2::from_shape_vec((shape[0], shape[1]), matrix_vec)
                .map_err(|e| SciRS2Error::ArrayError(format!("Array reshape failed: {}", e)))?;

            svd_f64_lapack(&arr, full_matrices)
                .map_err(|e| SciRS2Error::ComputationError(format!("SVD failed: {}", e)))
        })
        .await
        .map_err(|e| SciRS2Error::RuntimeError(format!("Task join error: {}", e)))??;

        Python::with_gil(|py| {
            use pyo3::types::PyDict;
            let dict = PyDict::new(py);
            dict.set_item("U", result.0.into_pyarray(py))?;
            dict.set_item("S", result.1.into_pyarray(py))?;
            dict.set_item("Vt", result.2.into_pyarray(py))?;
            Ok(dict.into_any())
        })
    })
}

/// Async QR decomposition for large matrices
#[pyfunction]
pub fn qr_async<'py>(
    py: Python<'py>,
    matrix: &Bound<'_, PyArray2<f64>>,
) -> PyResult<Bound<'py, PyAny>> {
    let matrix_vec: Vec<f64> = {
        let binding = matrix.readonly();
        let arr = binding.as_array();
        arr.to_vec()
    };
    let shape = matrix.shape();

    pyo3_asyncio::tokio::future_into_py(py, async move {
        let result = tokio::task::spawn_blocking(move || {
            use scirs2_linalg::qr_f64_lapack;
            use scirs2_core::Array2;

            let arr = Array2::from_shape_vec((shape[0], shape[1]), matrix_vec)
                .map_err(|e| SciRS2Error::ArrayError(format!("Array reshape failed: {}", e)))?;

            qr_f64_lapack(&arr)
                .map_err(|e| SciRS2Error::ComputationError(format!("QR failed: {}", e)))
        })
        .await
        .map_err(|e| SciRS2Error::RuntimeError(format!("Task join error: {}", e)))??;

        Python::with_gil(|py| {
            use pyo3::types::PyDict;
            let dict = PyDict::new(py);
            dict.set_item("Q", result.0.into_pyarray(py))?;
            dict.set_item("R", result.1.into_pyarray(py))?;
            Ok(dict.into_any())
        })
    })
}

/// Async numerical integration for expensive integrands
#[pyfunction]
pub fn quad_async<'py>(
    py: Python<'py>,
    func: PyObject,
    a: f64,
    b: f64,
    epsabs: Option<f64>,
    epsrel: Option<f64>,
) -> PyResult<Bound<'py, PyAny>> {
    pyo3_asyncio::tokio::future_into_py(py, async move {
        let result = tokio::task::spawn_blocking(move || {
            Python::with_gil(|py| {
                use scirs2_integrate::quad;

                // Create Rust closure that calls Python function
                let integrand = |x: f64| -> Result<f64, String> {
                    func.call1(py, (x,))
                        .and_then(|result| result.extract::<f64>(py))
                        .map_err(|e| format!("Python function call failed: {}", e))
                };

                let result = quad(
                    integrand,
                    a,
                    b,
                    epsabs.unwrap_or(1e-8),
                    epsrel.unwrap_or(1e-8),
                )
                .map_err(|e| SciRS2Error::ComputationError(format!("Integration failed: {}", e)))?;

                Ok((result.value, result.error))
            })
        })
        .await
        .map_err(|e| SciRS2Error::RuntimeError(format!("Task join error: {}", e)))??;

        Python::with_gil(|py| {
            use pyo3::types::PyDict;
            let dict = PyDict::new(py);
            dict.set_item("value", result.0)?;
            dict.set_item("error", result.1)?;
            Ok(dict.into_any())
        })
    })
}

/// Async optimization for expensive objective functions
#[pyfunction]
pub fn minimize_async<'py>(
    py: Python<'py>,
    func: PyObject,
    x0: &Bound<'_, PyArray1<f64>>,
    method: Option<String>,
    maxiter: Option<usize>,
) -> PyResult<Bound<'py, PyAny>> {
    let x0_vec: Vec<f64> = {
        let binding = x0.readonly();
        let arr = binding.as_array();
        arr.to_vec()
    };

    pyo3_asyncio::tokio::future_into_py(py, async move {
        let result = tokio::task::spawn_blocking(move || {
            Python::with_gil(|py| {
                use scirs2_optimize::{minimize, OptimizationMethod};

                // Create Rust closure that calls Python function
                let objective = |x: &[f64]| -> Result<f64, String> {
                    let x_py = pyo3::types::PyList::new(py, x)?;
                    func.call1(py, (x_py,))
                        .and_then(|result| result.extract::<f64>(py))
                        .map_err(|e| format!("Python function call failed: {}", e))
                };

                let method = match method.as_deref() {
                    Some("BFGS") => OptimizationMethod::BFGS,
                    Some("Newton") => OptimizationMethod::Newton,
                    Some("GradientDescent") => OptimizationMethod::GradientDescent,
                    _ => OptimizationMethod::BFGS,
                };

                let result = minimize(
                    objective,
                    &x0_vec,
                    method,
                    maxiter.unwrap_or(1000),
                )
                .map_err(|e| SciRS2Error::ComputationError(format!("Optimization failed: {}", e)))?;

                Ok((result.x, result.fun, result.nit))
            })
        })
        .await
        .map_err(|e| SciRS2Error::RuntimeError(format!("Task join error: {}", e)))??;

        Python::with_gil(|py| {
            use pyo3::types::PyDict;
            use scirs2_core::Array1;

            let dict = PyDict::new(py);
            let x = Array1::from_vec(result.0);
            dict.set_item("x", x.into_pyarray(py))?;
            dict.set_item("fun", result.1)?;
            dict.set_item("nit", result.2)?;
            Ok(dict.into_any())
        })
    })
}

/// Register async operations with Python module
pub fn register_async_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(fft_async, m)?)?;
    m.add_function(wrap_pyfunction!(svd_async, m)?)?;
    m.add_function(wrap_pyfunction!(qr_async, m)?)?;
    m.add_function(wrap_pyfunction!(quad_async, m)?)?;
    m.add_function(wrap_pyfunction!(minimize_async, m)?)?;
    Ok(())
}
