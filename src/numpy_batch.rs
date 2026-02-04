// Numpy-compatible batch processing for zero-copy transfer
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use rayon::prelude::*;

/// Process batch and return only aggregated statistics (no per-item Python objects)
#[pyfunction]
pub fn aggregate_stats(
    _py: Python<'_>,
    inputs: Vec<String>,
    pattern: &str,
) -> PyResult<(usize, usize, f64)> {
    let pat_bytes = pattern.as_bytes();
    if pat_bytes.is_empty() {
        return Ok((0, inputs.len(), 0.0));
    }
    let pat_len = pat_bytes.len();
    let first = pat_bytes[0];

    let total_len: usize = inputs.par_iter().map(|s| s.len()).sum();

    let matches: usize = inputs
        .par_iter()
        .filter(|input| {
            let bytes = input.as_bytes();
            bytes.len() >= pat_len && bytes[0] == first && &bytes[..pat_len] == pat_bytes
        })
        .count();

    let avg_len = total_len as f64 / inputs.len() as f64;

    Ok((matches, inputs.len(), avg_len))
}

/// Return raw bytes that can be viewed as numpy array (zero-copy)
#[pyfunction]
pub fn match_to_bytes<'py>(
    py: Python<'py>,
    inputs: Vec<String>,
    pattern: &str,
) -> PyResult<Bound<'py, PyBytes>> {
    let pat_bytes = pattern.as_bytes();
    let pat_len = pat_bytes.len();
    let first = pat_bytes[0];

    // Create byte array of results (0 or 1)
    let results: Vec<u8> = inputs
        .par_iter()
        .map(|input| {
            let bytes = input.as_bytes();
            if bytes.len() >= pat_len && bytes[0] == first && &bytes[..pat_len] == pat_bytes {
                1u8
            } else {
                0u8
            }
        })
        .collect();

    Ok(PyBytes::new(py, &results))
}

/// Process and return indices of matches only (sparse result)
#[pyfunction]
pub fn match_indices(_py: Python<'_>, inputs: Vec<String>, pattern: &str) -> PyResult<Vec<usize>> {
    let pat_bytes = pattern.as_bytes();
    let pat_len = pat_bytes.len();
    let first = pat_bytes[0];

    let indices: Vec<usize> = inputs
        .par_iter()
        .enumerate()
        .filter_map(|(idx, input)| {
            let bytes = input.as_bytes();
            if bytes.len() >= pat_len && bytes[0] == first && &bytes[..pat_len] == pat_bytes {
                Some(idx)
            } else {
                None
            }
        })
        .collect();

    Ok(indices)
}

/// Ultra-compact: Return match count and first N matches
#[pyfunction]
pub fn compact_results(
    _py: Python<'_>,
    inputs: Vec<String>,
    pattern: &str,
    max_results: usize,
) -> PyResult<(usize, Vec<String>)> {
    let pat_bytes = pattern.as_bytes();
    let pat_len = pat_bytes.len();
    let first = pat_bytes[0];

    let mut matches = Vec::with_capacity(max_results);
    let mut count = 0;

    for input in inputs {
        let bytes = input.as_bytes();
        if bytes.len() >= pat_len && bytes[0] == first && &bytes[..pat_len] == pat_bytes {
            count += 1;
            if matches.len() < max_results {
                matches.push(input);
            }
        }
    }

    Ok((count, matches))
}

/// Histogram of input lengths (for analysis)
#[pyfunction]
pub fn length_histogram(_py: Python<'_>, inputs: Vec<String>, bins: usize) -> PyResult<Vec<usize>> {
    let max_len = inputs.iter().map(|s| s.len()).max().unwrap_or(0);
    let bin_size = (max_len + bins) / bins;

    let mut histogram = vec![0; bins];

    for input in inputs {
        let bin = (input.len() / bin_size).min(bins - 1);
        histogram[bin] += 1;
    }

    Ok(histogram)
}

/// Streaming batch - process chunks without full materialization
#[pyfunction]
pub fn streaming_batch_count(
    _py: Python<'_>,
    inputs: Vec<String>,
    pattern: &str,
    chunk_size: usize,
) -> PyResult<usize> {
    let pat_bytes = pattern.as_bytes();
    let pat_len = pat_bytes.len();
    let first = pat_bytes[0];

    let count: usize = inputs
        .chunks(chunk_size)
        .map(|chunk| {
            chunk
                .iter()
                .filter(|input| {
                    let bytes = input.as_bytes();
                    bytes.len() >= pat_len && bytes[0] == first && &bytes[..pat_len] == pat_bytes
                })
                .count()
        })
        .sum();

    Ok(count)
}
