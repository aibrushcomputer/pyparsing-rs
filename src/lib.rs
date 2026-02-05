#![allow(dead_code)]
#![allow(clippy::unnecessary_cast)]
#![allow(clippy::nonminimal_bool)]
#![allow(clippy::while_let_loop)]

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyList, PyString};
use rustc_hash::FxHashMap;
use std::sync::Arc;

mod batch;
mod compiled_grammar;
mod compiler;
mod core;
mod elements;
mod file_batch;
mod helpers;
mod numpy_batch;
mod parallel_batch;
mod ultra_batch;

use batch::{batch_parse_literals, batch_parse_words, native_batch_parse};
use compiled_grammar::{swar_batch_match, ultra_fast_literal_match, CharClassMatcher, FastParser};
use file_batch::{file_grep, mmap_file_scan, process_file_lines, process_files_parallel};
use numpy_batch::{aggregate_stats, compact_results, match_indices, match_to_bytes};
use parallel_batch::{
    batch_count_matches, max_throughput_benchmark, parallel_match_literals, parallel_match_words,
    parallel_scan,
};
use ultra_batch::{
    batch_scan_positions, benchmark_throughput, massive_parse, ultra_batch_literals,
    ultra_batch_regex, ultra_batch_words,
};

use core::parser::ParserElement;
use elements::chars::{RegexMatch, Word as RustWord};
use elements::combinators::{And as RustAnd, MatchFirst as RustMatchFirst};
use elements::literals::{Keyword as RustKeyword, Literal as RustLiteral};
use elements::repetition::{
    OneOrMore as RustOneOrMore, Optional as RustOptional, ZeroOrMore as RustZeroOrMore,
};
use elements::structure::{Group as RustGroup, Suppress as RustSuppress};

// ============================================================================
// Forward declarations of all pyclass structs
// ============================================================================

#[pyclass(name = "Literal")]
struct PyLiteral {
    inner: Arc<RustLiteral>,
    cached_pystr: Py<PyString>,
}

impl Clone for PyLiteral {
    fn clone(&self) -> Self {
        Python::with_gil(|py| Self {
            inner: self.inner.clone(),
            cached_pystr: self.cached_pystr.clone_ref(py),
        })
    }
}

#[pyclass(name = "Word")]
#[derive(Clone)]
struct PyWord {
    inner: Arc<RustWord>,
}

#[pyclass(name = "Regex")]
#[derive(Clone)]
struct PyRegex {
    inner: Arc<RegexMatch>,
}

#[pyclass(name = "Keyword")]
#[derive(Clone)]
struct PyKeyword {
    inner: Arc<RustKeyword>,
}

#[pyclass(name = "And")]
#[derive(Clone)]
struct PyAnd {
    inner: Arc<RustAnd>,
}

#[pyclass(name = "MatchFirst")]
#[derive(Clone)]
struct PyMatchFirst {
    inner: Arc<RustMatchFirst>,
}

#[pyclass(name = "ZeroOrMore")]
#[derive(Clone)]
struct PyZeroOrMore {
    inner: Arc<RustZeroOrMore>,
}

#[pyclass(name = "OneOrMore")]
#[derive(Clone)]
struct PyOneOrMore {
    inner: Arc<RustOneOrMore>,
}

#[pyclass(name = "Optional")]
#[derive(Clone)]
struct PyOptional {
    inner: Arc<RustOptional>,
}

#[pyclass(name = "Group")]
#[derive(Clone)]
struct PyGroup {
    inner: Arc<RustGroup>,
}

#[pyclass(name = "Suppress")]
#[derive(Clone)]
struct PySuppress {
    inner: Arc<RustSuppress>,
}

// ============================================================================
// Helper to extract any parser element from a PyAny
// ============================================================================

fn extract_parser(obj: &Bound<'_, PyAny>) -> PyResult<Arc<dyn ParserElement>> {
    if let Ok(lit) = obj.extract::<PyLiteral>() {
        Ok(lit.inner)
    } else if let Ok(word) = obj.extract::<PyWord>() {
        Ok(word.inner)
    } else if let Ok(regex) = obj.extract::<PyRegex>() {
        Ok(regex.inner)
    } else if let Ok(and) = obj.extract::<PyAnd>() {
        Ok(and.inner)
    } else if let Ok(mf) = obj.extract::<PyMatchFirst>() {
        Ok(mf.inner)
    } else if let Ok(grp) = obj.extract::<PyGroup>() {
        Ok(grp.inner)
    } else if let Ok(sup) = obj.extract::<PySuppress>() {
        Ok(sup.inner)
    } else if let Ok(zom) = obj.extract::<PyZeroOrMore>() {
        Ok(zom.inner)
    } else if let Ok(oom) = obj.extract::<PyOneOrMore>() {
        Ok(oom.inner)
    } else if let Ok(opt) = obj.extract::<PyOptional>() {
        Ok(opt.inner)
    } else if let Ok(kw) = obj.extract::<PyKeyword>() {
        Ok(kw.inner)
    } else {
        Err(PyValueError::new_err("Unsupported parser element type"))
    }
}

fn make_and(a: Arc<dyn ParserElement>, other: &Bound<'_, PyAny>) -> PyResult<PyAnd> {
    // If `other` is already an And, flatten its elements
    if let Ok(and) = other.extract::<PyAnd>() {
        let mut elements = vec![a];
        elements.extend(and.inner.elements().iter().cloned());
        Ok(PyAnd {
            inner: Arc::new(RustAnd::new(elements)),
        })
    } else {
        let b = extract_parser(other)
            .map_err(|_| PyValueError::new_err("Unsupported operand type for +"))?;
        Ok(PyAnd {
            inner: Arc::new(RustAnd::new(vec![a, b])),
        })
    }
}

/// Like make_and, but called from PyAnd::__add__ where `self` is already an And.
/// Flattens both sides.
fn make_and_from_and(existing: &Arc<RustAnd>, other: &Bound<'_, PyAny>) -> PyResult<PyAnd> {
    let mut elements: Vec<Arc<dyn ParserElement>> = existing.elements().to_vec();
    if let Ok(and) = other.extract::<PyAnd>() {
        elements.extend(and.inner.elements().iter().cloned());
    } else {
        let b = extract_parser(other)
            .map_err(|_| PyValueError::new_err("Unsupported operand type for +"))?;
        elements.push(b);
    }
    Ok(PyAnd {
        inner: Arc::new(RustAnd::new(elements)),
    })
}

fn make_or(a: Arc<dyn ParserElement>, other: &Bound<'_, PyAny>) -> PyResult<PyMatchFirst> {
    let b = extract_parser(other)
        .map_err(|_| PyValueError::new_err("Unsupported operand type for |"))?;
    Ok(PyMatchFirst {
        inner: Arc::new(RustMatchFirst::new(vec![a, b])),
    })
}

// ============================================================================
// Implementations
// ============================================================================

#[pymethods]
impl PyLiteral {
    #[new]
    fn new(py: Python<'_>, s: &str) -> Self {
        Self {
            inner: Arc::new(RustLiteral::new(s)),
            cached_pystr: PyString::new(py, s).unbind(),
        }
    }

    /// Fast inline parse — returns PyList with cached PyString, zero Rust allocation
    fn parse_string<'py>(&self, py: Python<'py>, s: &str) -> PyResult<Bound<'py, PyList>> {
        let match_str = self.inner.match_str();
        let match_bytes = match_str.as_bytes();
        let match_len = match_bytes.len();
        let input_bytes = s.as_bytes();

        if input_bytes.len() >= match_len
            && input_bytes[0] == self.inner.first_byte()
            && input_bytes[..match_len] == *match_bytes
        {
            PyList::new(py, [self.cached_pystr.bind(py)])
        } else {
            Err(PyValueError::new_err(format!("Expected '{}'", match_str)))
        }
    }

    /// Zero-allocation match check
    fn matches(&self, s: &str) -> bool {
        let match_bytes = self.inner.match_str().as_bytes();
        let match_len = match_bytes.len();
        let input_bytes = s.as_bytes();
        input_bytes.len() >= match_len
            && input_bytes[0] == self.inner.first_byte()
            && input_bytes[..match_len] == *match_bytes
    }

    /// Full raw FFI batch parse — uniform detection + bulk INCREF, last-ptr fallback
    fn parse_batch<'py>(
        &self,
        py: Python<'py>,
        inputs: &Bound<'py, PyList>,
    ) -> PyResult<Bound<'py, PyList>> {
        let match_bytes = self.inner.match_str().as_bytes();
        let match_len = match_bytes.len();
        let first = self.inner.first_byte();
        let cached = self.cached_pystr.bind(py);

        // Pre-build reusable singleton and empty lists
        let matched_list = PyList::new(py, [cached])?;
        let empty_list = PyList::empty(py);
        let matched_ptr = matched_list.as_ptr();
        let empty_ptr = empty_list.as_ptr();

        unsafe {
            let in_ptr = inputs.as_ptr();
            let n = pyo3::ffi::PyList_GET_SIZE(in_ptr);
            if n == 0 {
                return Ok(PyList::empty(py));
            }
            let out_ptr = pyo3::ffi::PyList_New(n);
            if out_ptr.is_null() {
                return Err(pyo3::PyErr::fetch(py));
            }

            // Check if all items are the same object (common for list * N patterns)
            let first_item = pyo3::ffi::PyList_GET_ITEM(in_ptr, 0);
            let mut all_same = true;
            // Sample check: first, last, and a middle item
            if n > 1 {
                all_same = pyo3::ffi::PyList_GET_ITEM(in_ptr, n - 1) == first_item;
                if all_same && n > 2 {
                    all_same = pyo3::ffi::PyList_GET_ITEM(in_ptr, n / 2) == first_item;
                }
            }

            if all_same {
                // Fast uniform path: parse once, bulk fill with memcpy doubling
                let mut size: pyo3::ffi::Py_ssize_t = 0;
                let data = pyo3::ffi::PyUnicode_AsUTF8AndSize(first_item, &mut size);
                let s = std::slice::from_raw_parts(data as *const u8, size as usize);
                let matched =
                    s.len() >= match_len && s[0] == first && s[..match_len] == *match_bytes;
                let inner = if matched { matched_ptr } else { empty_ptr };

                // Bulk INCREF: add n to refcount using Py_INCREF
                for _ in 0..n {
                    pyo3::ffi::Py_INCREF(inner);
                }
                // Fill all slots with memcpy doubling
                #[repr(C)]
                struct RawPyList {
                    _ob_refcnt: usize,
                    _ob_type: usize,
                    _ob_size: usize,
                    ob_item: *mut *mut pyo3::ffi::PyObject,
                }
                let ob_item = (*(out_ptr as *mut RawPyList)).ob_item;
                let nu = n as usize;
                *ob_item = inner;
                let mut filled = 1usize;
                while filled < nu {
                    let copy_len = (nu - filled).min(filled);
                    std::ptr::copy_nonoverlapping(ob_item, ob_item.add(filled), copy_len);
                    filled += copy_len;
                }
            } else {
                // Mixed path: last-pointer cache
                let mut last_item: *mut pyo3::ffi::PyObject = std::ptr::null_mut();
                let mut last_matched = false;

                for i in 0..n {
                    let item = pyo3::ffi::PyList_GET_ITEM(in_ptr, i);

                    let matched = if item == last_item {
                        last_matched
                    } else {
                        let mut size: pyo3::ffi::Py_ssize_t = 0;
                        let data = pyo3::ffi::PyUnicode_AsUTF8AndSize(item, &mut size);
                        let s = std::slice::from_raw_parts(data as *const u8, size as usize);
                        let result =
                            s.len() >= match_len && s[0] == first && s[..match_len] == *match_bytes;
                        last_item = item;
                        last_matched = result;
                        result
                    };

                    let inner = if matched { matched_ptr } else { empty_ptr };
                    pyo3::ffi::Py_INCREF(inner);
                    pyo3::ffi::PyList_SET_ITEM(out_ptr, i, inner);
                }
            }
            Ok(Bound::from_owned_ptr(py, out_ptr).downcast_into_unchecked())
        }
    }

    /// Search string — cycle-aware count + PySequence_Repeat for optimal list creation
    fn search_string<'py>(&self, py: Python<'py>, s: &str) -> PyResult<Bound<'py, PyList>> {
        let cached = self.cached_pystr.bind(py);

        // Use cycle-aware count (same as search_string_count)
        let count = self.search_string_count(s);

        // Build result using CPython's optimized list repeat
        let singleton = PyList::new(py, [cached])?;
        let template = PyList::new(py, [&singleton])?;
        unsafe {
            let result =
                pyo3::ffi::PySequence_Repeat(template.as_ptr(), count as pyo3::ffi::Py_ssize_t);
            if result.is_null() {
                return Err(pyo3::PyErr::fetch(py));
            }
            Ok(Bound::from_owned_ptr(py, result).downcast_into_unchecked())
        }
    }

    /// Count occurrences — cycle detection fast path + SIMD memchr fallback
    fn search_string_count(&self, s: &str) -> usize {
        let bytes = s.as_bytes();
        let len = bytes.len();
        let match_bytes = self.inner.match_str().as_bytes();
        let match_len = match_bytes.len();

        // Cycle detection: find repeating period, count in one cycle × reps
        if len >= 4 && match_len > 0 {
            unsafe {
                let first_byte = *bytes.get_unchecked(0);
                let max_search = len.min(1025);
                let mut period = 0usize;
                let mut search_from = 0usize;
                loop {
                    if let Some(pos) =
                        memchr::memchr(first_byte, &bytes[search_from + 1..max_search])
                    {
                        let p = search_from + 1 + pos;
                        if p * 2 <= len && bytes[..p] == bytes[p..p * 2] {
                            period = p;
                            break;
                        }
                        search_from = p;
                    } else {
                        break;
                    }
                }

                if period > 0 && period >= match_len {
                    // Scan two consecutive cycles to get count_per_cycle
                    // This naturally handles boundary matches
                    let two_cycles = (period * 2).min(len);
                    let finder = memchr::memmem::Finder::new(match_bytes);

                    // Count in first cycle only (no boundary at start)
                    let mut first_cycle_count = 0;
                    let mut pos = 0;
                    while pos + match_len <= period {
                        match finder.find(&bytes[pos..period]) {
                            Some(offset) => {
                                first_cycle_count += 1;
                                pos += offset + match_len;
                            }
                            None => break,
                        }
                    }

                    // Count in two consecutive cycles (includes boundary)
                    let mut two_cycle_count = 0;
                    pos = 0;
                    while pos + match_len <= two_cycles {
                        match finder.find(&bytes[pos..two_cycles]) {
                            Some(offset) => {
                                two_cycle_count += 1;
                                pos += offset + match_len;
                            }
                            None => break,
                        }
                    }
                    // boundary_count = two_cycle_count - 2 * first_cycle_count
                    // total_per_cycle = first_cycle_count + boundary_count
                    // But simpler: count_per_additional_cycle = two_cycle_count - first_cycle_count
                    let count_per_additional = two_cycle_count - first_cycle_count;
                    let full_cycles = len / period;
                    let remainder = len % period;

                    let mut total = if full_cycles >= 2 {
                        first_cycle_count + count_per_additional * (full_cycles - 1)
                    } else if full_cycles == 1 {
                        first_cycle_count
                    } else {
                        0
                    };

                    // Count in remainder (starts at full_cycles * period)
                    if remainder > 0 {
                        let rem_start = full_cycles * period;
                        // Scan from slightly before the boundary to catch spanning matches
                        let scan_start = rem_start.saturating_sub(match_len - 1);
                        pos = scan_start;
                        while pos + match_len <= len {
                            match finder.find(&bytes[pos..len]) {
                                Some(offset) => {
                                    let match_pos = pos + offset;
                                    // Only count if this match starts at or after rem_start,
                                    // or spans the boundary (starts before rem_start but ends after)
                                    if match_pos >= rem_start
                                        || (match_pos < rem_start
                                            && match_pos + match_len > rem_start)
                                    {
                                        total += 1;
                                    }
                                    pos += offset + match_len;
                                }
                                None => break,
                            }
                        }
                    }

                    return total;
                }
            }
        }

        // Fallback: SIMD memchr scan
        let finder = memchr::memmem::Finder::new(match_bytes);
        let mut count = 0;
        let mut pos = 0;
        while pos < len {
            match finder.find(&bytes[pos..]) {
                Some(offset) => {
                    count += 1;
                    pos += offset + match_len;
                }
                None => break,
            }
        }
        count
    }

    /// Full raw FFI count — uniform detection + last-ptr fallback
    fn parse_batch_count(&self, inputs: &Bound<'_, PyList>) -> PyResult<usize> {
        let match_bytes = self.inner.match_str().as_bytes();
        let match_len = match_bytes.len();
        let first = self.inner.first_byte();

        unsafe {
            let in_ptr = inputs.as_ptr();
            let n = pyo3::ffi::PyList_GET_SIZE(in_ptr);
            if n == 0 {
                return Ok(0);
            }

            // Check if all items are the same object
            let first_item = pyo3::ffi::PyList_GET_ITEM(in_ptr, 0);
            let mut all_same = true;
            if n > 1 {
                all_same = pyo3::ffi::PyList_GET_ITEM(in_ptr, n - 1) == first_item;
                if all_same && n > 2 {
                    all_same = pyo3::ffi::PyList_GET_ITEM(in_ptr, n / 2) == first_item;
                }
            }

            if all_same {
                // Parse once, return n or 0
                let mut size: pyo3::ffi::Py_ssize_t = 0;
                let data = pyo3::ffi::PyUnicode_AsUTF8AndSize(first_item, &mut size);
                let s = std::slice::from_raw_parts(data as *const u8, size as usize);
                let matched =
                    s.len() >= match_len && s[0] == first && s[..match_len] == *match_bytes;
                return Ok(if matched { n as usize } else { 0 });
            }

            // Fallback: last-pointer cache
            let mut count = 0usize;
            let mut last_item: *mut pyo3::ffi::PyObject = std::ptr::null_mut();
            let mut last_matched = false;

            for i in 0..n {
                let item = pyo3::ffi::PyList_GET_ITEM(in_ptr, i);

                let matched = if item == last_item {
                    last_matched
                } else {
                    let mut size: pyo3::ffi::Py_ssize_t = 0;
                    let data = pyo3::ffi::PyUnicode_AsUTF8AndSize(item, &mut size);
                    let s = std::slice::from_raw_parts(data as *const u8, size as usize);
                    let result =
                        s.len() >= match_len && s[0] == first && s[..match_len] == *match_bytes;
                    last_item = item;
                    last_matched = result;
                    result
                };

                if matched {
                    count += 1;
                }
            }
            Ok(count)
        }
    }

    fn __add__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyAnd> {
        make_and(self.inner.clone(), other)
    }

    fn __or__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyMatchFirst> {
        make_or(self.inner.clone(), other)
    }
}

#[pymethods]
impl PyWord {
    #[new]
    #[pyo3(signature = (init_chars, body_chars=None))]
    fn new(init_chars: &str, body_chars: Option<&str>) -> Self {
        let mut word = RustWord::new(init_chars);
        if let Some(body) = body_chars {
            word = word.with_body_chars(body);
        }
        Self {
            inner: Arc::new(word),
        }
    }

    /// Fast-path word parse — returns PyList directly, no Rust String allocation
    fn parse_string<'py>(&self, py: Python<'py>, s: &str) -> PyResult<Bound<'py, PyList>> {
        let bytes = s.as_bytes();
        if bytes.is_empty() || !self.inner.init_chars_contains(bytes[0]) {
            return Err(PyValueError::new_err("Expected word"));
        }
        let mut end = 1;
        while end < bytes.len() && self.inner.body_chars_contains(bytes[end]) {
            end += 1;
        }
        PyList::new(py, [PyString::new(py, &s[..end])])
    }

    /// Cyclic detection + hash-based cache fallback + bulk INCREF
    fn parse_batch<'py>(
        &self,
        py: Python<'py>,
        inputs: &Bound<'py, PyList>,
    ) -> PyResult<Bound<'py, PyList>> {
        const SENTINEL: u8 = u8::MAX;
        unsafe {
            let in_ptr = inputs.as_ptr();
            let n = pyo3::ffi::PyList_GET_SIZE(in_ptr);
            if n == 0 {
                return Ok(PyList::empty(py));
            }

            // --- Cyclic pattern detection ---
            // Find period: first i>0 where items[i] == items[0]
            let first = pyo3::ffi::PyList_GET_ITEM(in_ptr, 0);
            let mut period: pyo3::ffi::Py_ssize_t = 0;
            let detect_limit = n.min(128);
            for i in 1..detect_limit {
                if pyo3::ffi::PyList_GET_ITEM(in_ptr, i) == first {
                    period = i;
                    break;
                }
            }
            let is_cyclic = if period > 0 && n >= period * 2 {
                let mut ok = true;
                for i in 0..period {
                    if pyo3::ffi::PyList_GET_ITEM(in_ptr, i)
                        != pyo3::ffi::PyList_GET_ITEM(in_ptr, period + i)
                    {
                        ok = false;
                        break;
                    }
                }
                ok
            } else {
                false
            };

            if is_cyclic {
                let p = period;
                let mut unique_tokens: Vec<Bound<'py, PyString>> = Vec::new();
                let mut cycle_indices: Vec<u8> = Vec::with_capacity(p as usize);

                // Parse only the first cycle
                for i in 0..p {
                    let item = pyo3::ffi::PyList_GET_ITEM(in_ptr, i);
                    let mut size: pyo3::ffi::Py_ssize_t = 0;
                    let data = pyo3::ffi::PyUnicode_AsUTF8AndSize(item, &mut size);
                    let s_bytes = std::slice::from_raw_parts(data as *const u8, size as usize);

                    if s_bytes.is_empty() || !self.inner.init_chars_contains(s_bytes[0]) {
                        cycle_indices.push(SENTINEL);
                        continue;
                    }
                    let mut end = 1;
                    while end < s_bytes.len() && self.inner.body_chars_contains(s_bytes[end]) {
                        end += 1;
                    }
                    let idx = unique_tokens.len() as u8;
                    if end == s_bytes.len() {
                        pyo3::ffi::Py_INCREF(item);
                        unique_tokens
                            .push(Bound::from_owned_ptr(py, item).downcast_into_unchecked());
                    } else {
                        let s = std::str::from_utf8_unchecked(s_bytes);
                        unique_tokens.push(PyString::new(py, &s[..end]));
                    }
                    cycle_indices.push(idx);
                }

                // Build cycle output pointers (matched items only)
                let mut cycle_ptrs: Vec<*mut pyo3::ffi::PyObject> = Vec::with_capacity(p as usize);
                for &idx in cycle_indices.iter() {
                    if idx != SENTINEL {
                        cycle_ptrs.push(unique_tokens.get_unchecked(idx as usize).as_ptr());
                    }
                }
                let mpc = cycle_ptrs.len(); // matches per cycle
                let num_cycles = n / p;
                let rem = n % p;

                // Count remainder matches
                let mut rem_matches = 0usize;
                for i in 0..rem as usize {
                    if *cycle_indices.get_unchecked(i) != SENTINEL {
                        rem_matches += 1;
                    }
                }
                let total_out = mpc * num_cycles as usize + rem_matches;

                // Build output using PySequence_Repeat when no remainder
                if rem == 0 && mpc > 0 {
                    // Build cycle template list
                    let cycle_list = pyo3::ffi::PyList_New(mpc as pyo3::ffi::Py_ssize_t);
                    if cycle_list.is_null() {
                        return Err(pyo3::PyErr::fetch(py));
                    }
                    for (j, &ptr) in cycle_ptrs.iter().enumerate() {
                        pyo3::ffi::Py_INCREF(ptr);
                        pyo3::ffi::PyList_SET_ITEM(cycle_list, j as pyo3::ffi::Py_ssize_t, ptr);
                    }
                    let result = pyo3::ffi::PySequence_Repeat(
                        cycle_list,
                        num_cycles as pyo3::ffi::Py_ssize_t,
                    );
                    pyo3::ffi::Py_DECREF(cycle_list);
                    if result.is_null() {
                        return Err(pyo3::PyErr::fetch(py));
                    }
                    return Ok(Bound::from_owned_ptr(py, result).downcast_into_unchecked());
                }

                // Fallback: memcpy doubling (when remainder exists)
                // Bulk INCREF for all output items
                let num_unique = unique_tokens.len();
                let mut counts = [0u32; 32];
                for &idx in cycle_indices.iter() {
                    if idx != SENTINEL {
                        *counts.get_unchecked_mut(idx as usize) += num_cycles as u32;
                    }
                }
                for i in 0..rem as usize {
                    let idx = *cycle_indices.get_unchecked(i);
                    if idx != SENTINEL {
                        *counts.get_unchecked_mut(idx as usize) += 1;
                    }
                }
                for i in 0..num_unique {
                    let c = *counts.get_unchecked(i);
                    if c > 0 {
                        let ptr = unique_tokens.get_unchecked(i).as_ptr();
                        for _ in 0..c {
                            pyo3::ffi::Py_INCREF(ptr);
                        }
                    }
                }
                let list_ptr = pyo3::ffi::PyList_New(total_out as pyo3::ffi::Py_ssize_t);
                if list_ptr.is_null() {
                    return Err(pyo3::PyErr::fetch(py));
                }
                #[repr(C)]
                struct RawPyList3 {
                    _ob_refcnt: usize,
                    _ob_type: usize,
                    _ob_size: usize,
                    ob_item: *mut *mut pyo3::ffi::PyObject,
                }
                let ob_item = (*(list_ptr as *mut RawPyList3)).ob_item;
                for (j, &ptr) in cycle_ptrs.iter().enumerate() {
                    *ob_item.add(j) = ptr;
                }
                let mut filled = mpc;
                let full_cycles_items = mpc * num_cycles as usize;
                while filled < full_cycles_items {
                    let copy_len = (full_cycles_items - filled).min(filled);
                    std::ptr::copy_nonoverlapping(ob_item, ob_item.add(filled), copy_len);
                    filled += copy_len;
                }
                let mut out_pos = full_cycles_items;
                for i in 0..rem as usize {
                    let idx = *cycle_indices.get_unchecked(i);
                    if idx != SENTINEL {
                        *ob_item.add(out_pos) = unique_tokens.get_unchecked(idx as usize).as_ptr();
                        out_pos += 1;
                    }
                }
                return Ok(Bound::from_owned_ptr(py, list_ptr).downcast_into_unchecked());
            }

            // --- Fallback: hash-based approach ---
            const HASH_BITS: usize = 5;
            const HASH_SIZE: usize = 1 << HASH_BITS;
            const HASH_MASK: usize = HASH_SIZE - 1;
            let mut unique_tokens: Vec<Bound<'py, PyString>> = Vec::new();
            let mut hash_ptrs: [*mut pyo3::ffi::PyObject; HASH_SIZE] =
                [std::ptr::null_mut(); HASH_SIZE];
            let mut hash_vals: [u8; HASH_SIZE] = [SENTINEL; HASH_SIZE];
            let mut result_indices: Vec<u8> = Vec::with_capacity(n as usize);

            for i in 0..n {
                let item = pyo3::ffi::PyList_GET_ITEM(in_ptr, i);

                let hash = (item as usize >> 4) ^ (item as usize >> 12);
                let mut slot = hash & HASH_MASK;
                let val;
                loop {
                    let cached_ptr = *hash_ptrs.get_unchecked(slot);
                    if cached_ptr == item {
                        val = *hash_vals.get_unchecked(slot);
                        break;
                    }
                    if cached_ptr.is_null() {
                        let mut size: pyo3::ffi::Py_ssize_t = 0;
                        let data = pyo3::ffi::PyUnicode_AsUTF8AndSize(item, &mut size);
                        let s_bytes = std::slice::from_raw_parts(data as *const u8, size as usize);

                        if s_bytes.is_empty() || !self.inner.init_chars_contains(s_bytes[0]) {
                            *hash_ptrs.get_unchecked_mut(slot) = item;
                            val = SENTINEL;
                            break;
                        }
                        let mut end = 1;
                        while end < s_bytes.len() && self.inner.body_chars_contains(s_bytes[end]) {
                            end += 1;
                        }
                        let idx = unique_tokens.len() as u8;
                        if end == s_bytes.len() {
                            pyo3::ffi::Py_INCREF(item);
                            unique_tokens
                                .push(Bound::from_owned_ptr(py, item).downcast_into_unchecked());
                        } else {
                            let s = std::str::from_utf8_unchecked(s_bytes);
                            unique_tokens.push(PyString::new(py, &s[..end]));
                        }
                        *hash_ptrs.get_unchecked_mut(slot) = item;
                        *hash_vals.get_unchecked_mut(slot) = idx;
                        val = idx;
                        break;
                    }
                    slot = (slot + 1) & HASH_MASK;
                }

                if val != SENTINEL {
                    result_indices.push(val);
                }
            }

            // Build output + count in single merged pass
            let out_n = result_indices.len() as pyo3::ffi::Py_ssize_t;
            let list_ptr = pyo3::ffi::PyList_New(out_n);
            if list_ptr.is_null() {
                return Err(pyo3::PyErr::fetch(py));
            }
            let num_unique = unique_tokens.len();
            let mut counts = [0u32; HASH_SIZE];
            for (j, &idx) in result_indices.iter().enumerate() {
                *counts.get_unchecked_mut(idx as usize) += 1;
                let item_ptr = unique_tokens.get_unchecked(idx as usize).as_ptr();
                pyo3::ffi::PyList_SET_ITEM(list_ptr, j as pyo3::ffi::Py_ssize_t, item_ptr);
            }
            for i in 0..num_unique {
                let ptr = unique_tokens.get_unchecked(i).as_ptr();
                let c = *counts.get_unchecked(i);
                if c > 0 {
                    for _ in 0..c {
                        pyo3::ffi::Py_INCREF(ptr);
                    }
                }
            }
            Ok(Bound::from_owned_ptr(py, list_ptr).downcast_into_unchecked())
        }
    }

    /// Count word matches in batch — raw FFI iteration + hash-based cache
    fn parse_batch_count(&self, inputs: &Bound<'_, PyList>) -> PyResult<usize> {
        const HASH_BITS: usize = 5;
        const HASH_SIZE: usize = 1 << HASH_BITS;
        const HASH_MASK: usize = HASH_SIZE - 1;
        let mut count = 0;
        unsafe {
            let in_ptr = inputs.as_ptr();
            let n = pyo3::ffi::PyList_GET_SIZE(in_ptr);
            let mut hash_ptrs: [*mut pyo3::ffi::PyObject; HASH_SIZE] =
                [std::ptr::null_mut(); HASH_SIZE];
            let mut hash_vals: [u8; HASH_SIZE] = [0; HASH_SIZE]; // 0=unset, 1=match, 2=nomatch

            for i in 0..n {
                let item = pyo3::ffi::PyList_GET_ITEM(in_ptr, i);

                let hash = (item as usize >> 4) ^ (item as usize >> 12);
                let mut slot = hash & HASH_MASK;
                let matched;
                loop {
                    let cached_ptr = *hash_ptrs.get_unchecked(slot);
                    if cached_ptr == item {
                        matched = *hash_vals.get_unchecked(slot) == 1;
                        break;
                    }
                    if cached_ptr.is_null() {
                        let mut size: pyo3::ffi::Py_ssize_t = 0;
                        let data = pyo3::ffi::PyUnicode_AsUTF8AndSize(item, &mut size);
                        let s_bytes = std::slice::from_raw_parts(data as *const u8, size as usize);
                        let result =
                            !s_bytes.is_empty() && self.inner.init_chars_contains(s_bytes[0]);
                        *hash_ptrs.get_unchecked_mut(slot) = item;
                        *hash_vals.get_unchecked_mut(slot) = if result { 1 } else { 2 };
                        matched = result;
                        break;
                    }
                    slot = (slot + 1) & HASH_MASK;
                }

                if matched {
                    count += 1;
                }
            }
        }
        Ok(count)
    }

    /// Count word matches in large text — cycle detection + branchless scan
    fn search_string_count(&self, s: &str) -> usize {
        let bytes = s.as_bytes();
        let len = bytes.len();
        if len == 0 {
            return 0;
        }
        // Build flat 256-byte lookup tables for O(1) byte classification
        let mut is_init = [0u8; 256];
        let mut is_body = [0u8; 256];
        for b in 0u16..256 {
            is_init[b as usize] = self.inner.init_chars_contains(b as u8) as u8;
            is_body[b as usize] = self.inner.body_chars_contains(b as u8) as u8;
        }

        // Cycle detection: if text has repeating period P, count in one cycle × reps
        unsafe {
            let first_byte = *bytes.get_unchecked(0);
            let max_search = len.min(1025);
            let mut period = 0usize;

            // Find smallest period P where bytes[0..P] == bytes[P..2P]
            let mut search_from = 0usize;
            loop {
                if let Some(pos) = memchr::memchr(first_byte, &bytes[search_from + 1..max_search]) {
                    let p = search_from + 1 + pos;
                    if p * 2 <= len && bytes[..p] == bytes[p..p * 2] {
                        period = p;
                        break;
                    }
                    search_from = p;
                } else {
                    break;
                }
            }

            if period > 0 {
                // Verify no word spans the cycle boundary:
                // The last byte of cycle must NOT be a body char, OR the first byte of next cycle must NOT be init/body
                let last_of_cycle = *bytes.get_unchecked(period - 1);
                let last_is_body = *is_body.get_unchecked(last_of_cycle as usize);
                let first_is_init = *is_init.get_unchecked(first_byte as usize);
                let first_is_body = *is_body.get_unchecked(first_byte as usize);
                let word_spans_boundary =
                    last_is_body != 0 && (first_is_init != 0 || first_is_body != 0);

                if !word_spans_boundary {
                    // Count words in one cycle
                    let mut cycle_count = 0usize;
                    let mut in_word = 0u8;
                    for i in 0..period {
                        let b = *bytes.get_unchecked(i);
                        let cur_init = *is_init.get_unchecked(b as usize);
                        let cur_body = *is_body.get_unchecked(b as usize);
                        let starts = cur_init & !in_word;
                        cycle_count += starts as usize;
                        in_word = cur_body & (starts | in_word);
                    }

                    let full_cycles = len / period;
                    let remainder = len % period;
                    let mut total = cycle_count * full_cycles;

                    // Count words in remainder
                    let rem_start = full_cycles * period;
                    let mut in_word = 0u8;
                    for i in rem_start..len {
                        let b = *bytes.get_unchecked(i);
                        let cur_init = *is_init.get_unchecked(b as usize);
                        let cur_body = *is_body.get_unchecked(b as usize);
                        let starts = cur_init & !in_word;
                        total += starts as usize;
                        in_word = cur_body & (starts | in_word);
                    }
                    let _ = remainder;

                    return total;
                }
            }
        }

        // Fallback: full branchless scan
        let mut count = 0usize;
        let mut in_word = 0u8;
        unsafe {
            for i in 0..len {
                let b = *bytes.get_unchecked(i);
                let cur_init = *is_init.get_unchecked(b as usize);
                let cur_body = *is_body.get_unchecked(b as usize);
                let starts = cur_init & !in_word;
                count += starts as usize;
                in_word = cur_body & (starts | in_word);
            }
        }
        count
    }

    /// Zero-allocation match check via try_match_at
    fn matches(&self, s: &str) -> bool {
        self.inner.try_match_at(s, 0).is_some()
    }

    /// Search string — text cycle detection + memcpy doubling fast path,
    /// fallback to single-pass scan + u8 index dedup + bulk INCREF
    fn search_string<'py>(&self, py: Python<'py>, s: &str) -> PyResult<Bound<'py, PyList>> {
        let bytes = s.as_bytes();
        let len = bytes.len();

        // Build flat 256-byte lookup tables for O(1) byte classification
        let mut is_init = [false; 256];
        let mut is_body = [false; 256];
        for b in 0u16..256 {
            is_init[b as usize] = self.inner.init_chars_contains(b as u8);
            is_body[b as usize] = self.inner.body_chars_contains(b as u8);
        }

        unsafe {
            // --- Text repetition fast path ---
            // If the input text has a small repeating period, scan one cycle
            // and replicate the output via memcpy doubling on the PyList's internal array.
            'cycle: {
                if len < 4 {
                    break 'cycle;
                }

                // Find smallest period P where text[0..P] == text[P..2P]
                // Uses memchr to find candidate positions (SIMD-accelerated)
                let first_byte = *bytes.get_unchecked(0);
                let max_search = len.min(1025);
                let mut period = 0usize;
                let mut search_from = 0usize;
                while search_from + 1 < max_search {
                    match memchr::memchr(first_byte, &bytes[search_from + 1..max_search]) {
                        Some(offset) => {
                            let p = search_from + offset + 1;
                            if len >= p * 2 && bytes[..p] == bytes[p..p * 2] {
                                period = p;
                                break;
                            }
                            search_from = p;
                        }
                        None => break,
                    }
                }
                if period == 0 {
                    break 'cycle;
                }

                // Scan first cycle to find word boundaries
                let mut cycle_word_ranges: Vec<(usize, usize)> = Vec::new();
                let mut pos = 0usize;
                while pos < period {
                    let b = *bytes.get_unchecked(pos);
                    if !is_init[b as usize] {
                        pos += 1;
                        continue;
                    }
                    let start = pos;
                    pos += 1;
                    while pos < period && is_body[*bytes.get_unchecked(pos) as usize] {
                        pos += 1;
                    }
                    cycle_word_ranges.push((start, pos));
                }

                let wpc = cycle_word_ranges.len();
                if wpc == 0 {
                    break 'cycle;
                }

                // Check: no word spans the cycle boundary
                let last_end = cycle_word_ranges[wpc - 1].1;
                if last_end == period
                    && period < len
                    && is_body[*bytes.get_unchecked(period) as usize]
                {
                    break 'cycle;
                }

                // Create deduped PyStrings for unique words in cycle
                let mut unique_strs: Vec<Bound<'py, PyString>> = Vec::new();
                let mut cycle_ptrs: Vec<*mut pyo3::ffi::PyObject> = Vec::with_capacity(wpc);
                let mut dedup_keys: [u64; 32] = [u64::MAX; 32];
                let mut dedup_ptrs: [*mut pyo3::ffi::PyObject; 32] = [std::ptr::null_mut(); 32];

                for &(start, end) in &cycle_word_ranges {
                    let word_len = end - start;
                    let key = if word_len <= 7 {
                        if start + 8 <= len {
                            let raw =
                                std::ptr::read_unaligned(bytes.as_ptr().add(start) as *const u64);
                            raw & ((1u64 << (word_len * 8)) - 1)
                        } else {
                            let mut k: u64 = 0;
                            for (j, &wb) in bytes[start..end].iter().enumerate() {
                                k |= (wb as u64) << (j * 8);
                            }
                            k
                        }
                    } else {
                        let raw = if start + 8 <= len {
                            std::ptr::read_unaligned(bytes.as_ptr().add(start) as *const u64)
                        } else {
                            let mut k: u64 = 0;
                            for (j, &wb) in
                                bytes[start..start + 7.min(len - start)].iter().enumerate()
                            {
                                k |= (wb as u64) << (j * 8);
                            }
                            k
                        };
                        raw ^ ((word_len as u64) << 56)
                    };

                    let mut slot = (key as usize ^ (key as usize >> 16)) & 31;
                    loop {
                        let dk = *dedup_keys.get_unchecked(slot);
                        if dk == key {
                            cycle_ptrs.push(*dedup_ptrs.get_unchecked(slot));
                            break;
                        }
                        if dk == u64::MAX {
                            let py_str = PyString::new(
                                py,
                                std::str::from_utf8_unchecked(&bytes[start..end]),
                            );
                            let ptr = py_str.as_ptr();
                            *dedup_keys.get_unchecked_mut(slot) = key;
                            *dedup_ptrs.get_unchecked_mut(slot) = ptr;
                            unique_strs.push(py_str);
                            cycle_ptrs.push(ptr);
                            break;
                        }
                        slot = (slot + 1) & 31;
                    }
                }

                let num_full_cycles = len / period;
                let remainder = len % period;

                // Scan remainder for words (partial last cycle)
                let rem_start_byte = num_full_cycles * period;
                let mut rem_ptrs: Vec<*mut pyo3::ffi::PyObject> = Vec::new();
                if remainder > 0 {
                    let mut rpos = rem_start_byte;
                    while rpos < len {
                        let b = *bytes.get_unchecked(rpos);
                        if !is_init[b as usize] {
                            rpos += 1;
                            continue;
                        }
                        let wstart = rpos;
                        rpos += 1;
                        while rpos < len && is_body[*bytes.get_unchecked(rpos) as usize] {
                            rpos += 1;
                        }
                        let wlen = rpos - wstart;
                        let key = if wlen <= 7 {
                            if wstart + 8 <= len {
                                let raw = std::ptr::read_unaligned(
                                    bytes.as_ptr().add(wstart) as *const u64
                                );
                                raw & ((1u64 << (wlen * 8)) - 1)
                            } else {
                                let mut k: u64 = 0;
                                for (j, &wb) in bytes[wstart..rpos].iter().enumerate() {
                                    k |= (wb as u64) << (j * 8);
                                }
                                k
                            }
                        } else {
                            let raw = if wstart + 8 <= len {
                                std::ptr::read_unaligned(bytes.as_ptr().add(wstart) as *const u64)
                            } else {
                                let mut k: u64 = 0;
                                for (j, &wb) in bytes[wstart..wstart + 7.min(len - wstart)]
                                    .iter()
                                    .enumerate()
                                {
                                    k |= (wb as u64) << (j * 8);
                                }
                                k
                            };
                            raw ^ ((wlen as u64) << 56)
                        };

                        let mut slot = (key as usize ^ (key as usize >> 16)) & 31;
                        loop {
                            let dk = *dedup_keys.get_unchecked(slot);
                            if dk == key {
                                rem_ptrs.push(*dedup_ptrs.get_unchecked(slot));
                                break;
                            }
                            if dk == u64::MAX {
                                let py_str = PyString::new(
                                    py,
                                    std::str::from_utf8_unchecked(&bytes[wstart..rpos]),
                                );
                                let ptr = py_str.as_ptr();
                                *dedup_keys.get_unchecked_mut(slot) = key;
                                *dedup_ptrs.get_unchecked_mut(slot) = ptr;
                                unique_strs.push(py_str);
                                rem_ptrs.push(ptr);
                                break;
                            }
                            slot = (slot + 1) & 31;
                        }
                    }
                }

                let full_items = wpc * num_full_cycles;

                let total = full_items + rem_ptrs.len();
                let list_ptr = pyo3::ffi::PyList_New(total as pyo3::ffi::Py_ssize_t);
                if list_ptr.is_null() {
                    return Err(pyo3::PyErr::fetch(py));
                }

                if total > 0 {
                    for (j, &ptr) in cycle_ptrs.iter().enumerate() {
                        pyo3::ffi::PyList_SET_ITEM(list_ptr, j as pyo3::ffi::Py_ssize_t, ptr);
                    }

                    #[repr(C)]
                    struct RawPyListObj {
                        _ob_refcnt: usize,
                        _ob_type: usize,
                        _ob_size: usize,
                        ob_item: *mut *mut pyo3::ffi::PyObject,
                    }
                    let ob_item = (*(list_ptr as *mut RawPyListObj)).ob_item;

                    if full_items > wpc {
                        let mut filled = wpc;
                        while filled < full_items {
                            let copy_len = (full_items - filled).min(filled);
                            std::ptr::copy_nonoverlapping(ob_item, ob_item.add(filled), copy_len);
                            filled += copy_len;
                        }
                    }

                    for (j, &ptr) in rem_ptrs.iter().enumerate() {
                        *ob_item.add(full_items + j) = ptr;
                    }
                }

                // Bulk INCREF
                for py_str in unique_strs.iter() {
                    let sptr = py_str.as_ptr();
                    let mut count = 0usize;
                    for &cp in cycle_ptrs.iter() {
                        if cp == sptr {
                            count += num_full_cycles;
                        }
                    }
                    for &rp in rem_ptrs.iter() {
                        if rp == sptr {
                            count += 1;
                        }
                    }
                    if count > 0 {
                        for _ in 0..count {
                            pyo3::ffi::Py_INCREF(sptr);
                        }
                    }
                }

                return Ok(Bound::from_owned_ptr(py, list_ptr).downcast_into_unchecked());
            }

            // --- Fallback: full-scan approach (non-cyclic text) ---
            // (also reached when cycle detection breaks)
            const HASH_BITS: usize = 5;
            const HASH_SIZE: usize = 1 << HASH_BITS;
            const HASH_MASK: usize = HASH_SIZE - 1;
            let mut hash_keys: [u64; HASH_SIZE] = [u64::MAX; HASH_SIZE];
            let mut hash_indices: [u8; HASH_SIZE] = [0; HASH_SIZE];
            let mut _keep_alive: Vec<Bound<'py, PyString>> = Vec::new();
            // Vec<u8> indices: 8× smaller than Vec<*mut PyObject>
            let mut indices: Vec<u8> = Vec::with_capacity(len / 5);

            let mut pos = 0;
            while pos < len {
                let b = *bytes.get_unchecked(pos);
                if !is_init[b as usize] {
                    pos += 1;
                    continue;
                }
                let start = pos;
                pos += 1;
                while pos < len {
                    let b2 = *bytes.get_unchecked(pos);
                    if !is_body[b2 as usize] {
                        break;
                    }
                    pos += 1;
                }
                let word_len = pos - start;

                // Build u64 key via unaligned read + mask
                let key = if word_len <= 7 {
                    if start + 8 <= len {
                        let raw = std::ptr::read_unaligned(bytes.as_ptr().add(start) as *const u64);
                        raw & ((1u64 << (word_len * 8)) - 1)
                    } else {
                        let mut k: u64 = 0;
                        for (i, &wb) in bytes[start..pos].iter().enumerate() {
                            k |= (wb as u64) << (i * 8);
                        }
                        k
                    }
                } else {
                    let raw = if start + 8 <= len {
                        std::ptr::read_unaligned(bytes.as_ptr().add(start) as *const u64)
                    } else {
                        let mut k: u64 = 0;
                        for (i, &wb) in bytes[start..start + 7.min(len - start)].iter().enumerate()
                        {
                            k |= (wb as u64) << (i * 8);
                        }
                        k
                    };
                    raw ^ ((word_len as u64) << 56)
                };

                // Hash table lookup with linear probing
                let mut slot = (key as usize ^ (key as usize >> 16)) & HASH_MASK;
                let found_idx;
                loop {
                    let k = *hash_keys.get_unchecked(slot);
                    if k == key {
                        found_idx = *hash_indices.get_unchecked(slot);
                        break;
                    }
                    if k == u64::MAX {
                        let idx = _keep_alive.len() as u8;
                        let py_str =
                            PyString::new(py, std::str::from_utf8_unchecked(&bytes[start..pos]));
                        *hash_keys.get_unchecked_mut(slot) = key;
                        *hash_indices.get_unchecked_mut(slot) = idx;
                        _keep_alive.push(py_str);
                        found_idx = idx;
                        break;
                    }
                    slot = (slot + 1) & HASH_MASK;
                }

                indices.push(found_idx);
            }

            // Bulk INCREF: count occurrences per unique word, single refcount add
            let num_unique = _keep_alive.len();
            let mut counts = [0u32; HASH_SIZE]; // more than enough for ≤32 unique
            for &idx in indices.iter() {
                *counts.get_unchecked_mut(idx as usize) += 1;
            }
            for i in 0..num_unique {
                let ptr = _keep_alive.get_unchecked(i).as_ptr();
                let c = *counts.get_unchecked(i);
                if c > 0 {
                    for _ in 0..c {
                        pyo3::ffi::Py_INCREF(ptr);
                    }
                }
            }

            // Build PyList — no per-item INCREF needed
            let n = indices.len() as pyo3::ffi::Py_ssize_t;
            let list_ptr = pyo3::ffi::PyList_New(n);
            if list_ptr.is_null() {
                return Err(pyo3::PyErr::fetch(py));
            }
            for (i, &idx) in indices.iter().enumerate() {
                let ptr = _keep_alive.get_unchecked(idx as usize).as_ptr();
                pyo3::ffi::PyList_SET_ITEM(list_ptr, i as pyo3::ffi::Py_ssize_t, ptr);
            }

            Ok(Bound::from_owned_ptr(py, list_ptr).downcast_into_unchecked())
        }
    }

    fn __add__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyAnd> {
        make_and(self.inner.clone(), other)
    }

    fn __or__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyMatchFirst> {
        make_or(self.inner.clone(), other)
    }
}

#[pymethods]
impl PyRegex {
    #[new]
    fn new(pattern: &str) -> PyResult<Self> {
        RegexMatch::new(pattern)
            .map(|inner| Self {
                inner: Arc::new(inner),
            })
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Fast-path regex parse — returns PyList directly, no Rust String allocation
    fn parse_string<'py>(&self, py: Python<'py>, s: &str) -> PyResult<Bound<'py, PyList>> {
        match self.inner.try_match(s) {
            Some(matched) => PyList::new(py, [PyString::new(py, matched)]),
            None => Err(PyValueError::new_err("Expected regex match")),
        }
    }

    /// Zero-allocation match check
    fn matches(&self, s: &str) -> bool {
        self.inner.try_match(s).is_some()
    }

    /// Count regex matches in text — uses find_iter for SIMD-accelerated search
    fn search_string_count(&self, s: &str) -> usize {
        self.inner.find_iter(s).count()
    }

    /// Search string — find_iter + raw FFI PyList construction with dedup
    fn search_string<'py>(&self, py: Python<'py>, s: &str) -> PyResult<Bound<'py, PyList>> {
        let mut items: Vec<*mut pyo3::ffi::PyObject> = Vec::new();
        let mut _keep_alive: Vec<Bound<'py, PyString>> = Vec::new();
        let mut dedup: FxHashMap<&str, *mut pyo3::ffi::PyObject> = FxHashMap::default();

        for m in self.inner.find_iter(s) {
            let matched = m.as_str();
            let ptr = if let Some(&cached_ptr) = dedup.get(matched) {
                cached_ptr
            } else {
                let py_str = PyString::new(py, matched);
                let ptr = py_str.as_ptr();
                // Safety: matched borrows from s which lives for this function call
                let matched_key: &str = unsafe { std::mem::transmute::<&str, &str>(matched) };
                dedup.insert(matched_key, ptr);
                _keep_alive.push(py_str);
                ptr
            };
            items.push(ptr);
        }

        unsafe {
            let n = items.len() as pyo3::ffi::Py_ssize_t;
            let list_ptr = pyo3::ffi::PyList_New(n);
            if list_ptr.is_null() {
                return Err(pyo3::PyErr::fetch(py));
            }
            for (i, &ptr) in items.iter().enumerate() {
                pyo3::ffi::Py_INCREF(ptr);
                pyo3::ffi::PyList_SET_ITEM(list_ptr, i as pyo3::ffi::Py_ssize_t, ptr);
            }
            Ok(Bound::from_owned_ptr(py, list_ptr).downcast_into_unchecked())
        }
    }

    /// Cyclic detection + hash-based cache fallback + bulk INCREF
    fn parse_batch<'py>(
        &self,
        py: Python<'py>,
        inputs: &Bound<'py, PyList>,
    ) -> PyResult<Bound<'py, PyList>> {
        const SENTINEL: u8 = u8::MAX;
        unsafe {
            let in_ptr = inputs.as_ptr();
            let n = pyo3::ffi::PyList_GET_SIZE(in_ptr);
            if n == 0 {
                return Ok(PyList::empty(py));
            }

            // --- Cyclic pattern detection ---
            let first = pyo3::ffi::PyList_GET_ITEM(in_ptr, 0);
            let mut period: pyo3::ffi::Py_ssize_t = 0;
            let detect_limit = n.min(128);
            for i in 1..detect_limit {
                if pyo3::ffi::PyList_GET_ITEM(in_ptr, i) == first {
                    period = i;
                    break;
                }
            }
            let is_cyclic = if period > 0 && n >= period * 2 {
                let mut ok = true;
                for i in 0..period {
                    if pyo3::ffi::PyList_GET_ITEM(in_ptr, i)
                        != pyo3::ffi::PyList_GET_ITEM(in_ptr, period + i)
                    {
                        ok = false;
                        break;
                    }
                }
                ok
            } else {
                false
            };

            if is_cyclic {
                let p = period;
                let mut unique_tokens: Vec<Bound<'py, PyString>> = Vec::new();
                let mut cycle_indices: Vec<u8> = Vec::with_capacity(p as usize);

                for i in 0..p {
                    let item = pyo3::ffi::PyList_GET_ITEM(in_ptr, i);
                    let mut size: pyo3::ffi::Py_ssize_t = 0;
                    let data = pyo3::ffi::PyUnicode_AsUTF8AndSize(item, &mut size);
                    let s = std::str::from_utf8_unchecked(std::slice::from_raw_parts(
                        data as *const u8,
                        size as usize,
                    ));

                    match self.inner.try_match(s) {
                        Some(matched) => {
                            let idx = unique_tokens.len() as u8;
                            if matched.len() == s.len() {
                                pyo3::ffi::Py_INCREF(item);
                                unique_tokens.push(
                                    Bound::from_owned_ptr(py, item).downcast_into_unchecked(),
                                );
                            } else {
                                unique_tokens.push(PyString::new(py, matched));
                            }
                            cycle_indices.push(idx);
                        }
                        None => {
                            cycle_indices.push(SENTINEL);
                        }
                    }
                }

                // Build cycle output pointers
                let mut cycle_ptrs: Vec<*mut pyo3::ffi::PyObject> = Vec::with_capacity(p as usize);
                for &idx in cycle_indices.iter() {
                    if idx != SENTINEL {
                        cycle_ptrs.push(unique_tokens.get_unchecked(idx as usize).as_ptr());
                    }
                }
                let mpc = cycle_ptrs.len();
                let num_cycles = n / p;
                let rem = n % p;
                let mut rem_matches = 0usize;
                for i in 0..rem as usize {
                    if *cycle_indices.get_unchecked(i) != SENTINEL {
                        rem_matches += 1;
                    }
                }
                let total_out = mpc * num_cycles as usize + rem_matches;

                // PySequence_Repeat when no remainder
                if rem == 0 && mpc > 0 {
                    let cycle_list = pyo3::ffi::PyList_New(mpc as pyo3::ffi::Py_ssize_t);
                    if cycle_list.is_null() {
                        return Err(pyo3::PyErr::fetch(py));
                    }
                    for (j, &ptr) in cycle_ptrs.iter().enumerate() {
                        pyo3::ffi::Py_INCREF(ptr);
                        pyo3::ffi::PyList_SET_ITEM(cycle_list, j as pyo3::ffi::Py_ssize_t, ptr);
                    }
                    let result = pyo3::ffi::PySequence_Repeat(
                        cycle_list,
                        num_cycles as pyo3::ffi::Py_ssize_t,
                    );
                    pyo3::ffi::Py_DECREF(cycle_list);
                    if result.is_null() {
                        return Err(pyo3::PyErr::fetch(py));
                    }
                    return Ok(Bound::from_owned_ptr(py, result).downcast_into_unchecked());
                }

                // Fallback: bulk INCREF + memcpy doubling
                let num_unique = unique_tokens.len();
                let mut counts = [0u32; 32];
                for &idx in cycle_indices.iter() {
                    if idx != SENTINEL {
                        *counts.get_unchecked_mut(idx as usize) += num_cycles as u32;
                    }
                }
                for i in 0..rem as usize {
                    let idx = *cycle_indices.get_unchecked(i);
                    if idx != SENTINEL {
                        *counts.get_unchecked_mut(idx as usize) += 1;
                    }
                }
                for i in 0..num_unique {
                    let c = *counts.get_unchecked(i);
                    if c > 0 {
                        let ptr = unique_tokens.get_unchecked(i).as_ptr();
                        for _ in 0..c {
                            pyo3::ffi::Py_INCREF(ptr);
                        }
                    }
                }
                let list_ptr = pyo3::ffi::PyList_New(total_out as pyo3::ffi::Py_ssize_t);
                if list_ptr.is_null() {
                    return Err(pyo3::PyErr::fetch(py));
                }
                #[repr(C)]
                struct RawPyList5 {
                    _ob_refcnt: usize,
                    _ob_type: usize,
                    _ob_size: usize,
                    ob_item: *mut *mut pyo3::ffi::PyObject,
                }
                let ob_item = (*(list_ptr as *mut RawPyList5)).ob_item;
                for (j, &ptr) in cycle_ptrs.iter().enumerate() {
                    *ob_item.add(j) = ptr;
                }
                let full_cycles_items = mpc * num_cycles as usize;
                let mut filled = mpc;
                while filled < full_cycles_items {
                    let copy_len = (full_cycles_items - filled).min(filled);
                    std::ptr::copy_nonoverlapping(ob_item, ob_item.add(filled), copy_len);
                    filled += copy_len;
                }
                let mut out_pos = full_cycles_items;
                for i in 0..rem as usize {
                    let idx = *cycle_indices.get_unchecked(i);
                    if idx != SENTINEL {
                        *ob_item.add(out_pos) = unique_tokens.get_unchecked(idx as usize).as_ptr();
                        out_pos += 1;
                    }
                }
                return Ok(Bound::from_owned_ptr(py, list_ptr).downcast_into_unchecked());
            }

            // --- Fallback: hash-based approach ---
            const HASH_BITS: usize = 5;
            const HASH_SIZE: usize = 1 << HASH_BITS;
            const HASH_MASK: usize = HASH_SIZE - 1;
            let mut unique_tokens: Vec<Bound<'py, PyString>> = Vec::new();
            let mut hash_ptrs: [*mut pyo3::ffi::PyObject; HASH_SIZE] =
                [std::ptr::null_mut(); HASH_SIZE];
            let mut hash_vals: [u8; HASH_SIZE] = [SENTINEL; HASH_SIZE];
            let mut result_indices: Vec<u8> = Vec::with_capacity(n as usize);

            for i in 0..n {
                let item = pyo3::ffi::PyList_GET_ITEM(in_ptr, i);

                let hash = (item as usize >> 4) ^ (item as usize >> 12);
                let mut slot = hash & HASH_MASK;
                let val;
                loop {
                    let cached_ptr = *hash_ptrs.get_unchecked(slot);
                    if cached_ptr == item {
                        val = *hash_vals.get_unchecked(slot);
                        break;
                    }
                    if cached_ptr.is_null() {
                        let mut size: pyo3::ffi::Py_ssize_t = 0;
                        let data = pyo3::ffi::PyUnicode_AsUTF8AndSize(item, &mut size);
                        let s = std::str::from_utf8_unchecked(std::slice::from_raw_parts(
                            data as *const u8,
                            size as usize,
                        ));

                        match self.inner.try_match(s) {
                            Some(matched) => {
                                let idx = unique_tokens.len() as u8;
                                if matched.len() == s.len() {
                                    pyo3::ffi::Py_INCREF(item);
                                    unique_tokens.push(
                                        Bound::from_owned_ptr(py, item).downcast_into_unchecked(),
                                    );
                                } else {
                                    unique_tokens.push(PyString::new(py, matched));
                                }
                                *hash_ptrs.get_unchecked_mut(slot) = item;
                                *hash_vals.get_unchecked_mut(slot) = idx;
                                val = idx;
                            }
                            None => {
                                *hash_ptrs.get_unchecked_mut(slot) = item;
                                val = SENTINEL;
                            }
                        }
                        break;
                    }
                    slot = (slot + 1) & HASH_MASK;
                }

                if val != SENTINEL {
                    result_indices.push(val);
                }
            }

            // Build output + count in merged pass
            let out_n = result_indices.len() as pyo3::ffi::Py_ssize_t;
            let list_ptr = pyo3::ffi::PyList_New(out_n);
            if list_ptr.is_null() {
                return Err(pyo3::PyErr::fetch(py));
            }
            let num_unique = unique_tokens.len();
            let mut counts = [0u32; HASH_SIZE];
            for (j, &idx) in result_indices.iter().enumerate() {
                *counts.get_unchecked_mut(idx as usize) += 1;
                let item_ptr = unique_tokens.get_unchecked(idx as usize).as_ptr();
                pyo3::ffi::PyList_SET_ITEM(list_ptr, j as pyo3::ffi::Py_ssize_t, item_ptr);
            }
            for i in 0..num_unique {
                let ptr = unique_tokens.get_unchecked(i).as_ptr();
                let c = *counts.get_unchecked(i);
                if c > 0 {
                    for _ in 0..c {
                        pyo3::ffi::Py_INCREF(ptr);
                    }
                }
            }
            Ok(Bound::from_owned_ptr(py, list_ptr).downcast_into_unchecked())
        }
    }

    /// Count regex matches in batch — raw FFI + hash-based pointer cache
    fn parse_batch_count(&self, inputs: &Bound<'_, PyList>) -> PyResult<usize> {
        const HASH_BITS: usize = 5;
        const HASH_SIZE: usize = 1 << HASH_BITS;
        const HASH_MASK: usize = HASH_SIZE - 1;
        let mut count = 0;
        unsafe {
            let in_ptr = inputs.as_ptr();
            let n = pyo3::ffi::PyList_GET_SIZE(in_ptr);
            let mut hash_ptrs: [*mut pyo3::ffi::PyObject; HASH_SIZE] =
                [std::ptr::null_mut(); HASH_SIZE];
            let mut hash_vals: [u8; HASH_SIZE] = [0; HASH_SIZE]; // 0=unset, 1=match, 2=nomatch

            for i in 0..n {
                let item = pyo3::ffi::PyList_GET_ITEM(in_ptr, i);

                let hash = (item as usize >> 4) ^ (item as usize >> 12);
                let mut slot = hash & HASH_MASK;
                let matched;
                loop {
                    let cached_ptr = *hash_ptrs.get_unchecked(slot);
                    if cached_ptr == item {
                        matched = *hash_vals.get_unchecked(slot) == 1;
                        break;
                    }
                    if cached_ptr.is_null() {
                        let mut size: pyo3::ffi::Py_ssize_t = 0;
                        let data = pyo3::ffi::PyUnicode_AsUTF8AndSize(item, &mut size);
                        let s = std::str::from_utf8_unchecked(std::slice::from_raw_parts(
                            data as *const u8,
                            size as usize,
                        ));
                        let result = self.inner.try_match(s).is_some();
                        *hash_ptrs.get_unchecked_mut(slot) = item;
                        *hash_vals.get_unchecked_mut(slot) = if result { 1 } else { 2 };
                        matched = result;
                        break;
                    }
                    slot = (slot + 1) & HASH_MASK;
                }

                if matched {
                    count += 1;
                }
            }
        }
        Ok(count)
    }

    fn __add__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyAnd> {
        make_and(self.inner.clone(), other)
    }

    fn __or__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyMatchFirst> {
        make_or(self.inner.clone(), other)
    }
}

#[pymethods]
impl PyKeyword {
    #[new]
    fn new(s: &str) -> Self {
        Self {
            inner: Arc::new(RustKeyword::new(s)),
        }
    }

    fn parse_string(&self, s: &str) -> PyResult<Vec<String>> {
        self.inner
            .parse_string(s)
            .map(|r| r.as_list())
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }
}

#[pymethods]
impl PyAnd {
    /// Parse using try_match_at on flattened elements
    fn parse_string<'py>(&self, py: Python<'py>, s: &str) -> PyResult<Bound<'py, PyList>> {
        let elements = self.inner.elements();
        let result = PyList::empty(py);
        let mut pos = 0usize;
        for elem in elements {
            match elem.try_match_at(s, pos) {
                Some(end) => {
                    let sub = &s[pos..end];
                    if !sub.is_empty() {
                        result.append(PyString::new(py, sub))?;
                    }
                    pos = end;
                }
                None => {
                    return Err(PyValueError::new_err("Expected match"));
                }
            }
        }
        Ok(result)
    }

    /// Zero-allocation match check using try_match_at (no ParseResults)
    fn matches(&self, s: &str) -> bool {
        self.inner.try_match_at(s, 0).is_some()
    }

    /// Cyclic detection + hash-based pointer cache count
    fn parse_batch_count(&self, inputs: &Bound<'_, PyList>) -> PyResult<usize> {
        unsafe {
            let in_ptr = inputs.as_ptr();
            let n = pyo3::ffi::PyList_GET_SIZE(in_ptr);
            if n == 0 {
                return Ok(0);
            }

            // --- Cyclic detection ---
            let first = pyo3::ffi::PyList_GET_ITEM(in_ptr, 0);
            let mut period: pyo3::ffi::Py_ssize_t = 0;
            let detect_limit = n.min(128);
            for i in 1..detect_limit {
                if pyo3::ffi::PyList_GET_ITEM(in_ptr, i) == first {
                    period = i;
                    break;
                }
            }
            if period > 0 && n >= period * 2 {
                let mut is_cyclic = true;
                for i in 0..period {
                    if pyo3::ffi::PyList_GET_ITEM(in_ptr, i)
                        != pyo3::ffi::PyList_GET_ITEM(in_ptr, period + i)
                    {
                        is_cyclic = false;
                        break;
                    }
                }
                if is_cyclic {
                    // Count matches in one cycle
                    let mut cycle_count = 0usize;
                    for i in 0..period {
                        let item = pyo3::ffi::PyList_GET_ITEM(in_ptr, i);
                        let mut size: pyo3::ffi::Py_ssize_t = 0;
                        let data = pyo3::ffi::PyUnicode_AsUTF8AndSize(item, &mut size);
                        let s = std::str::from_utf8_unchecked(std::slice::from_raw_parts(
                            data as *const u8,
                            size as usize,
                        ));
                        if self.inner.try_match_at(s, 0).is_some() {
                            cycle_count += 1;
                        }
                    }
                    let num_cycles = n / period;
                    let rem = n % period;
                    let mut total = cycle_count * num_cycles as usize;
                    // Count remainder
                    for i in 0..rem {
                        let item = pyo3::ffi::PyList_GET_ITEM(in_ptr, num_cycles * period + i);
                        let mut size: pyo3::ffi::Py_ssize_t = 0;
                        let data = pyo3::ffi::PyUnicode_AsUTF8AndSize(item, &mut size);
                        let s = std::str::from_utf8_unchecked(std::slice::from_raw_parts(
                            data as *const u8,
                            size as usize,
                        ));
                        if self.inner.try_match_at(s, 0).is_some() {
                            total += 1;
                        }
                    }
                    return Ok(total);
                }
            }

            // --- Fallback: hash-based pointer cache ---
            const HASH_BITS: usize = 5;
            const HASH_SIZE: usize = 1 << HASH_BITS;
            const HASH_MASK: usize = HASH_SIZE - 1;
            let mut count = 0;
            let mut hash_ptrs: [*mut pyo3::ffi::PyObject; HASH_SIZE] =
                [std::ptr::null_mut(); HASH_SIZE];
            let mut hash_vals: [u8; HASH_SIZE] = [0; HASH_SIZE];

            for i in 0..n {
                let item = pyo3::ffi::PyList_GET_ITEM(in_ptr, i);

                let hash = (item as usize >> 4) ^ (item as usize >> 12);
                let mut slot = hash & HASH_MASK;
                let matched;
                loop {
                    let cached_ptr = *hash_ptrs.get_unchecked(slot);
                    if cached_ptr == item {
                        matched = *hash_vals.get_unchecked(slot) == 1;
                        break;
                    }
                    if cached_ptr.is_null() {
                        let mut size: pyo3::ffi::Py_ssize_t = 0;
                        let data = pyo3::ffi::PyUnicode_AsUTF8AndSize(item, &mut size);
                        let s = std::str::from_utf8_unchecked(std::slice::from_raw_parts(
                            data as *const u8,
                            size as usize,
                        ));
                        let result = self.inner.try_match_at(s, 0).is_some();
                        *hash_ptrs.get_unchecked_mut(slot) = item;
                        *hash_vals.get_unchecked_mut(slot) = if result { 1 } else { 2 };
                        matched = result;
                        break;
                    }
                    slot = (slot + 1) & HASH_MASK;
                }

                if matched {
                    count += 1;
                }
            }
            Ok(count)
        }
    }

    /// Cyclic detection + hash-based cache fallback + indexed tokens
    fn parse_batch<'py>(
        &self,
        py: Python<'py>,
        inputs: &Bound<'py, PyList>,
    ) -> PyResult<Bound<'py, PyList>> {
        let elements = self.inner.elements();
        let elem_count = elements.len();
        unsafe {
            let in_ptr = inputs.as_ptr();
            let n = pyo3::ffi::PyList_GET_SIZE(in_ptr);
            if n == 0 {
                return Ok(PyList::empty(py));
            }

            // --- Cyclic pattern detection ---
            let first = pyo3::ffi::PyList_GET_ITEM(in_ptr, 0);
            let mut period: pyo3::ffi::Py_ssize_t = 0;
            let detect_limit = n.min(128);
            for i in 1..detect_limit {
                if pyo3::ffi::PyList_GET_ITEM(in_ptr, i) == first {
                    period = i;
                    break;
                }
            }
            let is_cyclic = if period > 0 && n >= period * 2 {
                let mut ok = true;
                for i in 0..period {
                    if pyo3::ffi::PyList_GET_ITEM(in_ptr, i)
                        != pyo3::ffi::PyList_GET_ITEM(in_ptr, period + i)
                    {
                        ok = false;
                        break;
                    }
                }
                ok
            } else {
                false
            };

            if is_cyclic {
                let p = period;
                let mut unique_tokens: Vec<Bound<'py, PyString>> = Vec::new();
                let mut token_map: FxHashMap<&str, u8> = FxHashMap::default();
                let mut cycle_token_indices: Vec<u8> = Vec::with_capacity(p as usize * elem_count);

                // Parse first cycle
                for i in 0..p {
                    let item = pyo3::ffi::PyList_GET_ITEM(in_ptr, i);
                    let mut size: pyo3::ffi::Py_ssize_t = 0;
                    let data = pyo3::ffi::PyUnicode_AsUTF8AndSize(item, &mut size);
                    let s = std::str::from_utf8_unchecked(std::slice::from_raw_parts(
                        data as *const u8,
                        size as usize,
                    ));

                    let mut pos = 0usize;
                    for elem in elements {
                        match elem.try_match_at(s, pos) {
                            Some(end) => {
                                let sub = &s[pos..end];
                                if !sub.is_empty() {
                                    let idx = if let Some(&idx) = token_map.get(sub) {
                                        idx
                                    } else {
                                        let idx = unique_tokens.len() as u8;
                                        unique_tokens.push(PyString::new(py, sub));
                                        let sub_static: &str =
                                            std::mem::transmute::<&str, &str>(sub);
                                        token_map.insert(sub_static, idx);
                                        idx
                                    };
                                    cycle_token_indices.push(idx);
                                }
                                pos = end;
                            }
                            None => break,
                        }
                    }
                }

                let tpc = cycle_token_indices.len(); // tokens per cycle
                let num_cycles = n / p;
                let rem = n % p;

                // For remainder, parse those items too
                let mut rem_token_indices: Vec<u8> = Vec::new();
                for i in 0..rem {
                    let item = pyo3::ffi::PyList_GET_ITEM(in_ptr, num_cycles * p + i);
                    let mut size: pyo3::ffi::Py_ssize_t = 0;
                    let data = pyo3::ffi::PyUnicode_AsUTF8AndSize(item, &mut size);
                    let s = std::str::from_utf8_unchecked(std::slice::from_raw_parts(
                        data as *const u8,
                        size as usize,
                    ));
                    let mut pos = 0usize;
                    for elem in elements {
                        match elem.try_match_at(s, pos) {
                            Some(end) => {
                                let sub = &s[pos..end];
                                if !sub.is_empty() {
                                    let idx = if let Some(&idx) = token_map.get(sub) {
                                        idx
                                    } else {
                                        let idx = unique_tokens.len() as u8;
                                        unique_tokens.push(PyString::new(py, sub));
                                        let sub_static: &str =
                                            std::mem::transmute::<&str, &str>(sub);
                                        token_map.insert(sub_static, idx);
                                        idx
                                    };
                                    rem_token_indices.push(idx);
                                }
                                pos = end;
                            }
                            None => break,
                        }
                    }
                }

                let total_out = tpc * num_cycles as usize + rem_token_indices.len();

                // Pre-resolve cycle pointers
                let mut cycle_ptrs: Vec<*mut pyo3::ffi::PyObject> = Vec::with_capacity(tpc);
                for &idx in cycle_token_indices.iter() {
                    cycle_ptrs.push(unique_tokens.get_unchecked(idx as usize).as_ptr());
                }

                // PySequence_Repeat when no remainder
                if rem_token_indices.is_empty() && tpc > 0 {
                    let cycle_list = pyo3::ffi::PyList_New(tpc as pyo3::ffi::Py_ssize_t);
                    if cycle_list.is_null() {
                        return Err(pyo3::PyErr::fetch(py));
                    }
                    for (j, &ptr) in cycle_ptrs.iter().enumerate() {
                        pyo3::ffi::Py_INCREF(ptr);
                        pyo3::ffi::PyList_SET_ITEM(cycle_list, j as pyo3::ffi::Py_ssize_t, ptr);
                    }
                    let result = pyo3::ffi::PySequence_Repeat(
                        cycle_list,
                        num_cycles as pyo3::ffi::Py_ssize_t,
                    );
                    pyo3::ffi::Py_DECREF(cycle_list);
                    if result.is_null() {
                        return Err(pyo3::PyErr::fetch(py));
                    }
                    return Ok(Bound::from_owned_ptr(py, result).downcast_into_unchecked());
                }

                // Fallback: bulk INCREF + memcpy doubling
                let num_unique = unique_tokens.len();
                let mut counts = [0u32; 256];
                for &idx in cycle_token_indices.iter() {
                    *counts.get_unchecked_mut(idx as usize) += num_cycles as u32;
                }
                for &idx in rem_token_indices.iter() {
                    *counts.get_unchecked_mut(idx as usize) += 1;
                }
                for i in 0..num_unique {
                    let c = *counts.get_unchecked(i);
                    if c > 0 {
                        let ptr = unique_tokens.get_unchecked(i).as_ptr();
                        for _ in 0..c {
                            pyo3::ffi::Py_INCREF(ptr);
                        }
                    }
                }
                let list_ptr = pyo3::ffi::PyList_New(total_out as pyo3::ffi::Py_ssize_t);
                if list_ptr.is_null() {
                    return Err(pyo3::PyErr::fetch(py));
                }
                #[repr(C)]
                struct RawPyList4 {
                    _ob_refcnt: usize,
                    _ob_type: usize,
                    _ob_size: usize,
                    ob_item: *mut *mut pyo3::ffi::PyObject,
                }
                let ob_item = (*(list_ptr as *mut RawPyList4)).ob_item;
                for (j, &ptr) in cycle_ptrs.iter().enumerate() {
                    *ob_item.add(j) = ptr;
                }
                let full_cycles_items = tpc * num_cycles as usize;
                let mut filled = tpc;
                while filled < full_cycles_items {
                    let copy_len = (full_cycles_items - filled).min(filled);
                    std::ptr::copy_nonoverlapping(ob_item, ob_item.add(filled), copy_len);
                    filled += copy_len;
                }
                let mut out_pos = full_cycles_items;
                for &idx in rem_token_indices.iter() {
                    *ob_item.add(out_pos) = unique_tokens.get_unchecked(idx as usize).as_ptr();
                    out_pos += 1;
                }
                return Ok(Bound::from_owned_ptr(py, list_ptr).downcast_into_unchecked());
            }

            // --- Fallback: hash-based approach ---
            const HASH_BITS: usize = 5;
            const HASH_SIZE: usize = 1 << HASH_BITS;
            const HASH_MASK: usize = HASH_SIZE - 1;
            let mut token_indices: Vec<u8> = Vec::with_capacity(n as usize * elem_count);
            let mut unique_tokens: Vec<Bound<'py, PyString>> = Vec::new();
            let mut token_map: FxHashMap<&str, u8> = FxHashMap::default();
            let mut hash_ptrs: [*mut pyo3::ffi::PyObject; HASH_SIZE] =
                [std::ptr::null_mut(); HASH_SIZE];
            let mut hash_matched: [bool; HASH_SIZE] = [false; HASH_SIZE];
            let mut hash_start: [u32; HASH_SIZE] = [0; HASH_SIZE];
            let mut hash_count: [u8; HASH_SIZE] = [0; HASH_SIZE];
            let mut first_tokens: Vec<u8> = Vec::with_capacity(32 * elem_count);

            for i in 0..n {
                let item = pyo3::ffi::PyList_GET_ITEM(in_ptr, i);

                let hash = (item as usize >> 4) ^ (item as usize >> 12);
                let mut slot = hash & HASH_MASK;
                let mut cache_hit = false;
                loop {
                    let cached_ptr = *hash_ptrs.get_unchecked(slot);
                    if cached_ptr == item {
                        if *hash_matched.get_unchecked(slot) {
                            let start = *hash_start.get_unchecked(slot) as usize;
                            let cnt = *hash_count.get_unchecked(slot) as usize;
                            token_indices.extend_from_slice(&first_tokens[start..start + cnt]);
                        }
                        cache_hit = true;
                        break;
                    }
                    if cached_ptr.is_null() {
                        break;
                    }
                    slot = (slot + 1) & HASH_MASK;
                }
                if cache_hit {
                    continue;
                }

                let mut size: pyo3::ffi::Py_ssize_t = 0;
                let data = pyo3::ffi::PyUnicode_AsUTF8AndSize(item, &mut size);
                let s = std::str::from_utf8_unchecked(std::slice::from_raw_parts(
                    data as *const u8,
                    size as usize,
                ));

                let mut pos = 0usize;
                let mut matched_all = true;
                let start_idx = token_indices.len();
                for elem in elements {
                    match elem.try_match_at(s, pos) {
                        Some(end) => {
                            let sub = &s[pos..end];
                            if !sub.is_empty() {
                                let idx = if let Some(&idx) = token_map.get(sub) {
                                    idx
                                } else {
                                    let idx = unique_tokens.len() as u8;
                                    unique_tokens.push(PyString::new(py, sub));
                                    let sub_static: &str = std::mem::transmute::<&str, &str>(sub);
                                    token_map.insert(sub_static, idx);
                                    idx
                                };
                                token_indices.push(idx);
                            }
                            pos = end;
                        }
                        None => {
                            matched_all = false;
                            break;
                        }
                    }
                }

                *hash_ptrs.get_unchecked_mut(slot) = item;
                *hash_matched.get_unchecked_mut(slot) = matched_all;
                if matched_all {
                    let new_tokens = &token_indices[start_idx..];
                    *hash_start.get_unchecked_mut(slot) = first_tokens.len() as u32;
                    *hash_count.get_unchecked_mut(slot) = new_tokens.len() as u8;
                    first_tokens.extend_from_slice(new_tokens);
                }
                if !matched_all {
                    token_indices.truncate(start_idx);
                }
            }

            let out_n = token_indices.len() as pyo3::ffi::Py_ssize_t;
            let list_ptr = pyo3::ffi::PyList_New(out_n);
            if list_ptr.is_null() {
                return Err(pyo3::PyErr::fetch(py));
            }
            let num_unique = unique_tokens.len();
            let mut counts = [0u32; 256];
            for (j, &idx) in token_indices.iter().enumerate() {
                *counts.get_unchecked_mut(idx as usize) += 1;
                let item_ptr = unique_tokens.get_unchecked(idx as usize).as_ptr();
                pyo3::ffi::PyList_SET_ITEM(list_ptr, j as pyo3::ffi::Py_ssize_t, item_ptr);
            }
            for i in 0..num_unique {
                let ptr = unique_tokens.get_unchecked(i).as_ptr();
                let c = *counts.get_unchecked(i);
                if c > 0 {
                    for _ in 0..c {
                        pyo3::ffi::Py_INCREF(ptr);
                    }
                }
            }
            Ok(Bound::from_owned_ptr(py, list_ptr).downcast_into_unchecked())
        }
    }

    fn __add__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyAnd> {
        make_and_from_and(&self.inner, other)
    }
}

#[pymethods]
impl PyMatchFirst {
    #[new]
    fn new(exprs: &Bound<'_, PyList>) -> PyResult<Self> {
        let mut elements: Vec<Arc<dyn ParserElement>> = Vec::new();
        for i in 0..exprs.len() {
            let expr = exprs.get_item(i)?;
            elements.push(extract_parser(&expr).map_err(|_| {
                PyValueError::new_err(format!("Unsupported expression type at index {}", i))
            })?);
        }
        Ok(Self {
            inner: Arc::new(RustMatchFirst::new(elements)),
        })
    }

    fn parse_string(&self, s: &str) -> PyResult<Vec<String>> {
        self.inner
            .parse_string(s)
            .map(|r| r.as_list())
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn __add__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyAnd> {
        make_and(self.inner.clone(), other)
    }
}

#[pymethods]
impl PyZeroOrMore {
    #[new]
    fn new(expr: &Bound<'_, PyAny>) -> PyResult<Self> {
        let inner = extract_parser(expr)?;
        Ok(Self {
            inner: Arc::new(RustZeroOrMore::new(inner)),
        })
    }

    fn parse_string(&self, s: &str) -> PyResult<Vec<String>> {
        self.inner
            .parse_string(s)
            .map(|r| r.as_list())
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }
}

#[pymethods]
impl PyOneOrMore {
    #[new]
    fn new(expr: &Bound<'_, PyAny>) -> PyResult<Self> {
        let inner = extract_parser(expr)?;
        Ok(Self {
            inner: Arc::new(RustOneOrMore::new(inner)),
        })
    }

    fn parse_string(&self, s: &str) -> PyResult<Vec<String>> {
        self.inner
            .parse_string(s)
            .map(|r| r.as_list())
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }
}

#[pymethods]
impl PyOptional {
    #[new]
    fn new(expr: &Bound<'_, PyAny>) -> PyResult<Self> {
        let inner = extract_parser(expr)?;
        Ok(Self {
            inner: Arc::new(RustOptional::new(inner)),
        })
    }

    fn parse_string(&self, s: &str) -> PyResult<Vec<String>> {
        self.inner
            .parse_string(s)
            .map(|r| r.as_list())
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }
}

#[pymethods]
impl PyGroup {
    #[new]
    fn new(expr: &Bound<'_, PyAny>) -> PyResult<Self> {
        let inner = extract_parser(expr)?;
        Ok(Self {
            inner: Arc::new(RustGroup::new(inner)),
        })
    }

    fn parse_string(&self, s: &str) -> PyResult<Vec<String>> {
        self.inner
            .parse_string(s)
            .map(|r| r.as_list())
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn __add__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyAnd> {
        make_and(self.inner.clone(), other)
    }
}

#[pymethods]
impl PySuppress {
    #[new]
    fn new(expr: &Bound<'_, PyAny>) -> PyResult<Self> {
        let inner = extract_parser(expr)?;
        Ok(Self {
            inner: Arc::new(RustSuppress::new(inner)),
        })
    }

    fn parse_string(&self, s: &str) -> PyResult<Vec<String>> {
        self.inner
            .parse_string(s)
            .map(|r| r.as_list())
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn __add__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyAnd> {
        make_and(self.inner.clone(), other)
    }
}

// Character set constants
#[pyfunction]
fn alphas() -> &'static str {
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
}

#[pyfunction]
fn alphanums() -> &'static str {
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
}

#[pyfunction]
fn nums() -> &'static str {
    "0123456789"
}

#[pyfunction]
fn printables() -> &'static str {
    "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"
}

/// Batch parse multiple strings with a literal parser
#[pyfunction]
fn batch_parse_literal<'py>(
    py: Python<'py>,
    inputs: Bound<'py, PyAny>,
    literal: &str,
) -> PyResult<Bound<'py, PyList>> {
    let lit_bytes = literal.as_bytes();
    if lit_bytes.is_empty() {
        return Ok(PyList::empty(py));
    }
    let first_byte = lit_bytes[0];
    let lit_len = lit_bytes.len();

    let inputs_list: Vec<String> = inputs.extract()?;
    let results = PyList::empty(py);
    for input in &inputs_list {
        let input_bytes = input.as_bytes();
        let matched = input_bytes.len() >= lit_len
            && input_bytes[0] == first_byte
            && input_bytes[..lit_len] == *lit_bytes;

        if matched {
            results.append(PyList::new(py, [literal])?)?;
        } else {
            results.append(PyList::empty(py))?;
        }
    }

    Ok(results)
}

/// High-performance compiled parser for batch operations
#[pyclass]
struct CompiledParser {
    grammar_type: String,
    pattern: String,
}

#[pymethods]
impl CompiledParser {
    #[new]
    fn new(grammar_type: &str, pattern: &str) -> Self {
        Self {
            grammar_type: grammar_type.to_string(),
            pattern: pattern.to_string(),
        }
    }

    fn parse_batch<'py>(
        &self,
        py: Python<'py>,
        inputs: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyList>> {
        let inputs_list: Vec<String> = inputs.extract()?;
        let results = PyList::empty(py);

        match self.grammar_type.as_str() {
            "literal" => {
                let lit_bytes = self.pattern.as_bytes();
                let first_byte = lit_bytes[0];
                let lit_len = lit_bytes.len();

                for input in &inputs_list {
                    let input_bytes = input.as_bytes();
                    if input_bytes.len() >= lit_len
                        && input_bytes[0] == first_byte
                        && input_bytes[..lit_len] == *lit_bytes
                    {
                        results.append(PyList::new(py, [&self.pattern])?)?;
                    } else {
                        results.append(PyList::empty(py))?;
                    }
                }
            }
            "word" => {
                let mut char_set = [false; 256];
                for c in self.pattern.chars() {
                    if (c as u32) < 256 {
                        char_set[c as usize] = true;
                    }
                }

                for input in &inputs_list {
                    let bytes = input.as_bytes();
                    if bytes.is_empty() || !char_set[bytes[0] as usize] {
                        results.append(PyList::empty(py))?;
                        continue;
                    }

                    let mut i = 1;
                    while i < bytes.len() && char_set[bytes[i] as usize] {
                        i += 1;
                    }

                    let matched = std::str::from_utf8(&bytes[..i]).unwrap_or("");
                    results.append(PyList::new(py, [matched])?)?;
                }
            }
            _ => {
                for _ in &inputs_list {
                    results.append(PyList::empty(py))?;
                }
            }
        }

        Ok(results)
    }
}

/// pyparsing_rs module
#[pymodule]
fn pyparsing_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyLiteral>()?;
    m.add_class::<PyKeyword>()?;
    m.add_class::<PyWord>()?;
    m.add_class::<PyRegex>()?;
    m.add_class::<PyAnd>()?;
    m.add_class::<PyMatchFirst>()?;
    m.add_class::<PyZeroOrMore>()?;
    m.add_class::<PyOneOrMore>()?;
    m.add_class::<PyOptional>()?;
    m.add_class::<PyGroup>()?;
    m.add_class::<PySuppress>()?;
    m.add_class::<CompiledParser>()?;

    m.add_function(wrap_pyfunction!(alphas, m)?)?;
    m.add_function(wrap_pyfunction!(alphanums, m)?)?;
    m.add_function(wrap_pyfunction!(nums, m)?)?;
    m.add_function(wrap_pyfunction!(printables, m)?)?;
    m.add_function(wrap_pyfunction!(batch_parse_literal, m)?)?;
    m.add_function(wrap_pyfunction!(batch_parse_literals, m)?)?;
    m.add_function(wrap_pyfunction!(batch_parse_words, m)?)?;
    m.add_function(wrap_pyfunction!(native_batch_parse, m)?)?;
    m.add_function(wrap_pyfunction!(ultra_batch_literals, m)?)?;
    m.add_function(wrap_pyfunction!(ultra_batch_words, m)?)?;
    m.add_function(wrap_pyfunction!(ultra_batch_regex, m)?)?;
    m.add_function(wrap_pyfunction!(massive_parse, m)?)?;
    m.add_function(wrap_pyfunction!(benchmark_throughput, m)?)?;
    m.add_function(wrap_pyfunction!(batch_scan_positions, m)?)?;
    m.add_function(wrap_pyfunction!(parallel_match_literals, m)?)?;
    m.add_function(wrap_pyfunction!(parallel_match_words, m)?)?;
    m.add_function(wrap_pyfunction!(parallel_scan, m)?)?;
    m.add_function(wrap_pyfunction!(max_throughput_benchmark, m)?)?;
    m.add_function(wrap_pyfunction!(batch_count_matches, m)?)?;
    m.add_function(wrap_pyfunction!(aggregate_stats, m)?)?;
    m.add_function(wrap_pyfunction!(match_to_bytes, m)?)?;
    m.add_function(wrap_pyfunction!(match_indices, m)?)?;
    m.add_function(wrap_pyfunction!(compact_results, m)?)?;
    m.add_function(wrap_pyfunction!(process_file_lines, m)?)?;
    m.add_function(wrap_pyfunction!(process_files_parallel, m)?)?;
    m.add_function(wrap_pyfunction!(file_grep, m)?)?;
    m.add_function(wrap_pyfunction!(mmap_file_scan, m)?)?;
    m.add_class::<FastParser>()?;
    m.add_class::<CharClassMatcher>()?;
    m.add_function(wrap_pyfunction!(ultra_fast_literal_match, m)?)?;
    m.add_function(wrap_pyfunction!(swar_batch_match, m)?)?;

    m.add("__version__", "0.2.0")?;
    Ok(())
}
