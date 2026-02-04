// ULTRA HIGH PERFORMANCE BATCH PARSING
// Process thousands of inputs in a single FFI call with zero Python overhead

use pyo3::prelude::*;
use pyo3::types::PyList;

/// Ultra-fast batch literal matching - processes 10K+ items per FFI call
#[pyfunction]
pub fn ultra_batch_literals(
    _py: Python<'_>,
    inputs: Vec<String>,
    literal: &str,
) -> PyResult<Vec<bool>> {
    let lit_bytes = literal.as_bytes();
    if lit_bytes.is_empty() {
        return Ok(vec![false; inputs.len()]);
    }
    let lit_len = lit_bytes.len();
    let first_byte = lit_bytes[0];

    let chunk_size = 1024;
    let mut results = Vec::with_capacity(inputs.len());

    for chunk in inputs.chunks(chunk_size) {
        for input in chunk {
            let bytes = input.as_bytes();
            let matched =
                bytes.len() >= lit_len && bytes[0] == first_byte && bytes[..lit_len] == *lit_bytes;
            results.push(matched);
        }
    }

    Ok(results)
}

/// Native word batch parsing with SIMD-style character classification
#[pyfunction]
pub fn ultra_batch_words(
    _py: Python<'_>,
    inputs: Vec<String>,
    init_chars: &str,
) -> PyResult<Vec<Option<String>>> {
    let mut lookup = [0u8; 256];
    for c in init_chars.chars() {
        if (c as u32) < 256 {
            lookup[c as usize] = 1;
        }
    }

    let mut results = Vec::with_capacity(inputs.len());

    for input in inputs {
        let bytes = input.as_bytes();
        if bytes.is_empty() || lookup[bytes[0] as usize] == 0 {
            results.push(None);
            continue;
        }

        let mut i = 1;
        while i < bytes.len() && lookup[bytes[i] as usize] != 0 {
            i += 1;
        }

        let word = unsafe { std::str::from_utf8_unchecked(&bytes[..i]) };
        results.push(Some(word.to_string()));
    }

    Ok(results)
}

/// Native regex batch matching using Rust's regex engine
#[pyfunction]
pub fn ultra_batch_regex(
    _py: Python<'_>,
    inputs: Vec<String>,
    pattern: &str,
) -> PyResult<Vec<Option<String>>> {
    use regex::Regex;

    let regex = Regex::new(pattern)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

    let mut results = Vec::with_capacity(inputs.len());

    for input in inputs {
        if let Some(mat) = regex.find(&input) {
            results.push(Some(mat.as_str().to_string()));
        } else {
            results.push(None);
        }
    }

    Ok(results)
}

/// Massive throughput parser - handles 100K+ items
#[pyfunction]
pub fn massive_parse<'py>(
    py: Python<'py>,
    inputs: Vec<String>,
    parser_type: &str,
    pattern: &str,
) -> PyResult<Bound<'py, PyList>> {
    let results = PyList::empty(py);

    match parser_type {
        "literal" => {
            let pat_bytes = pattern.as_bytes();
            let pat_len = pat_bytes.len();
            let first = pat_bytes[0];

            for input in inputs {
                let bytes = input.as_bytes();
                let matched =
                    bytes.len() >= pat_len && bytes[0] == first && bytes[..pat_len] == *pat_bytes;

                if matched {
                    results.append(pattern)?;
                } else {
                    results.append(py.None())?;
                }
            }
        }
        "word" => {
            let mut lookup = [false; 256];
            for c in pattern.chars() {
                if (c as u32) < 256 {
                    lookup[c as usize] = true;
                }
            }

            for input in inputs {
                let bytes = input.as_bytes();
                if bytes.is_empty() || !lookup[bytes[0] as usize] {
                    results.append(py.None())?;
                    continue;
                }

                let mut i = 1;
                while i < bytes.len() && lookup[bytes[i] as usize] {
                    i += 1;
                }

                let word = unsafe { std::str::from_utf8_unchecked(&bytes[..i]) };
                results.append(word)?;
            }
        }
        "digit" => {
            for input in inputs {
                let bytes = input.as_bytes();
                let mut start = None;
                let mut end = 0;

                for (i, &b) in bytes.iter().enumerate() {
                    if b.is_ascii_digit() {
                        if start.is_none() {
                            start = Some(i);
                        }
                        end = i + 1;
                    } else if start.is_some() {
                        break;
                    }
                }

                if let Some(s) = start {
                    let num = unsafe { std::str::from_utf8_unchecked(&bytes[s..end]) };
                    results.append(num)?;
                } else {
                    results.append(py.None())?;
                }
            }
        }
        _ => {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Unknown parser type: {}",
                parser_type
            )));
        }
    }

    Ok(results)
}

/// Benchmark throughput - returns ops/sec
#[pyfunction]
pub fn benchmark_throughput(
    _py: Python<'_>,
    inputs: Vec<String>,
    parser_type: &str,
    pattern: &str,
) -> PyResult<f64> {
    use std::time::Instant;

    let start = Instant::now();

    match parser_type {
        "literal" => {
            let pat_bytes = pattern.as_bytes();
            let pat_len = pat_bytes.len();
            let first = pat_bytes[0];

            for input in &inputs {
                let bytes = input.as_bytes();
                let _ =
                    bytes.len() >= pat_len && bytes[0] == first && bytes[..pat_len] == *pat_bytes;
            }
        }
        "word" => {
            let mut lookup = [false; 256];
            for c in pattern.chars() {
                if (c as u32) < 256 {
                    lookup[c as usize] = true;
                }
            }

            for input in &inputs {
                let bytes = input.as_bytes();
                if bytes.is_empty() || !lookup[bytes[0] as usize] {
                    continue;
                }
                let mut i = 1;
                while i < bytes.len() && lookup[bytes[i] as usize] {
                    i += 1;
                }
            }
        }
        _ => {}
    }

    let elapsed = start.elapsed().as_secs_f64();
    let ops_per_sec = inputs.len() as f64 / elapsed;

    Ok(ops_per_sec)
}

/// Zero-copy batch scanner - returns positions only
#[pyfunction]
pub fn batch_scan_positions(
    _py: Python<'_>,
    inputs: Vec<String>,
    pattern: &str,
) -> PyResult<Vec<Option<(usize, usize)>>> {
    let pat_bytes = pattern.as_bytes();
    let pat_len = pat_bytes.len();
    let mut results = Vec::with_capacity(inputs.len());

    for input in inputs {
        if let Some(pos) = memchr::memmem::find(input.as_bytes(), pat_bytes) {
            results.push(Some((pos, pos + pat_len)));
        } else {
            results.push(None);
        }
    }

    Ok(results)
}
