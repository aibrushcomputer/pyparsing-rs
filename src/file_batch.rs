// File-based batch processing - zero Python string overhead
use pyo3::prelude::*;
use rayon::prelude::*;
use std::fs::File;
use std::io::{BufRead, BufReader};

/// Process a file directly in Rust - no Python string creation
#[pyfunction]
pub fn process_file_lines(
    _py: Python<'_>,
    filepath: &str,
    pattern: &str,
) -> PyResult<(usize, usize)> {
    let file = File::open(filepath)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;

    let reader = BufReader::new(file);
    let pat_bytes = pattern.as_bytes();
    let pat_len = pat_bytes.len();
    let first = pat_bytes[0];

    let (total, matches): (usize, usize) = reader
        .lines()
        .par_bridge()
        .map(|line| {
            let line = line.unwrap_or_default();
            let bytes = line.as_bytes();
            let is_match =
                bytes.len() >= pat_len && bytes[0] == first && &bytes[..pat_len] == pat_bytes;
            (1, if is_match { 1 } else { 0 })
        })
        .reduce(|| (0, 0), |a, b| (a.0 + b.0, a.1 + b.1));

    Ok((total, matches))
}

/// Process multiple files in parallel
#[pyfunction]
pub fn process_files_parallel(
    _py: Python<'_>,
    filepaths: Vec<String>,
    pattern: &str,
) -> PyResult<Vec<(String, usize, usize)>> {
    let pat_bytes = pattern.as_bytes();
    let pat_len = pat_bytes.len();
    let first = pat_bytes[0];

    let results: Vec<(String, usize, usize)> = filepaths
        .par_iter()
        .map(|filepath| {
            let file = match File::open(filepath) {
                Ok(f) => f,
                Err(_) => return (filepath.clone(), 0, 0),
            };

            let reader = BufReader::new(file);
            let (total, matches) = reader
                .lines()
                .map(|line| {
                    let line = line.unwrap_or_default();
                    let bytes = line.as_bytes();
                    let is_match = bytes.len() >= pat_len
                        && bytes[0] == first
                        && &bytes[..pat_len] == pat_bytes;
                    (1, if is_match { 1 } else { 0 })
                })
                .fold((0, 0), |a, b| (a.0 + b.0, a.1 + b.1));

            (filepath.clone(), total, matches)
        })
        .collect();

    Ok(results)
}

/// Ultra-fast file grep - returns matching lines
#[pyfunction]
pub fn file_grep(
    _py: Python<'_>,
    filepath: &str,
    pattern: &str,
    max_results: usize,
) -> PyResult<Vec<String>> {
    let file = File::open(filepath)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;

    let reader = BufReader::new(file);
    let pat_bytes = pattern.as_bytes();

    let results: Vec<String> = reader
        .lines()
        .filter_map(|line| {
            let line = line.ok()?;
            if memchr::memmem::find(line.as_bytes(), pat_bytes).is_some() {
                Some(line)
            } else {
                None
            }
        })
        .take(max_results)
        .collect();

    Ok(results)
}

/// Memory-map file and process (ultra-fast for large files)
#[pyfunction]
pub fn mmap_file_scan(_py: Python<'_>, filepath: &str, pattern: &str) -> PyResult<usize> {
    use memmap2::Mmap;

    let file = File::open(filepath)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;

    let mmap = unsafe { Mmap::map(&file) }
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;

    let pat_bytes = pattern.as_bytes();

    // Count occurrences
    let count = memchr::memmem::find_iter(&mmap, pat_bytes).count();

    Ok(count)
}

/// Process CSV-like data in Rust
#[pyfunction]
pub fn process_csv_field(
    _py: Python<'_>,
    filepath: &str,
    delimiter: u8,
    field_index: usize,
    pattern: &str,
) -> PyResult<usize> {
    let file = File::open(filepath)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;

    let reader = BufReader::new(file);
    let pat_bytes = pattern.as_bytes();

    let count: usize = reader
        .lines()
        .par_bridge()
        .filter_map(|line| {
            let line = line.ok()?;
            let fields: Vec<&str> = line.split(|c| c as u8 == delimiter).collect();
            fields.get(field_index).and_then(|field| {
                if memchr::memmem::find(field.as_bytes(), pat_bytes).is_some() {
                    Some(1)
                } else {
                    None
                }
            })
        })
        .sum();

    Ok(count)
}

/// Large file split processing
#[pyfunction]
pub fn split_file_process(
    _py: Python<'_>,
    filepath: &str,
    pattern: &str,
    num_chunks: usize,
) -> PyResult<usize> {
    use memmap2::Mmap;

    let file = File::open(filepath)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;

    let mmap = unsafe { Mmap::map(&file) }
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;

    let pat_bytes = pattern.as_bytes();
    let len = mmap.len();
    let chunk_size = len / num_chunks;

    let count: usize = (0..num_chunks)
        .into_par_iter()
        .map(|i| {
            let start = i * chunk_size;
            let end = if i == num_chunks - 1 {
                len
            } else {
                (i + 1) * chunk_size
            };
            memchr::memmem::find_iter(&mmap[start..end], pat_bytes).count()
        })
        .sum();

    Ok(count)
}
