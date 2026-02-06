#![allow(clippy::unnecessary_cast)]
#![allow(clippy::nonminimal_bool)]
#![allow(clippy::while_let_loop)]

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyList, PyString};
use rustc_hash::FxHashMap;
use std::sync::Arc;

mod core;
mod elements;

use core::parser::ParserElement;
use elements::chars::{RegexMatch, Word as RustWord};
use elements::combinators::{And as RustAnd, MatchFirst as RustMatchFirst};
use elements::literals::{Keyword as RustKeyword, Literal as RustLiteral};
use elements::repetition::{
    OneOrMore as RustOneOrMore, Optional as RustOptional, ZeroOrMore as RustZeroOrMore,
};
use elements::structure::{Group as RustGroup, Suppress as RustSuppress};

// ============================================================================
// Raw FFI helpers — deduplicated from repeated inline patterns
// ============================================================================

/// Raw CPython PyListObject layout for direct ob_item access.
#[repr(C)]
struct RawPyList {
    _ob_refcnt: usize,
    _ob_type: usize,
    _ob_size: usize,
    ob_item: *mut *mut pyo3::ffi::PyObject,
}

/// Extract UTF-8 bytes from a Python string object (no allocation).
#[inline(always)]
unsafe fn py_str_as_bytes<'a>(obj: *mut pyo3::ffi::PyObject) -> &'a [u8] {
    let mut size: pyo3::ffi::Py_ssize_t = 0;
    let data = pyo3::ffi::PyUnicode_AsUTF8AndSize(obj, &mut size);
    std::slice::from_raw_parts(data as *const u8, size as usize)
}

/// Extract UTF-8 str from a Python string object (no allocation).
#[inline(always)]
unsafe fn py_str_as_str<'a>(obj: *mut pyo3::ffi::PyObject) -> &'a str {
    std::str::from_utf8_unchecked(py_str_as_bytes(obj))
}

/// Bulk increment reference count for a Python object.
#[inline(always)]
unsafe fn bulk_incref(ptr: *mut pyo3::ffi::PyObject, count: usize) {
    for _ in 0..count {
        pyo3::ffi::Py_INCREF(ptr);
    }
}

/// Access internal ob_item pointer of a PyList (raw FFI).
#[inline(always)]
unsafe fn list_ob_item(list_ptr: *mut pyo3::ffi::PyObject) -> *mut *mut pyo3::ffi::PyObject {
    (*(list_ptr as *mut RawPyList)).ob_item
}

/// Check if all items in a PyList point to the same Python object.
/// Uses direct ob_item access for cache-friendly contiguous memory scan.
#[inline(always)]
unsafe fn list_all_same(list_ptr: *mut pyo3::ffi::PyObject, n: pyo3::ffi::Py_ssize_t) -> bool {
    if n <= 1 {
        return true;
    }
    let ob_items = list_ob_item(list_ptr);
    let first = *ob_items;
    let mut i = 1usize;
    let nu = n as usize;
    // Process 4 pointers at a time for better ILP
    while i + 3 < nu {
        if *ob_items.add(i) != first
            || *ob_items.add(i + 1) != first
            || *ob_items.add(i + 2) != first
            || *ob_items.add(i + 3) != first
        {
            return false;
        }
        i += 4;
    }
    while i < nu {
        if *ob_items.add(i) != first {
            return false;
        }
        i += 1;
    }
    true
}

/// Build a u64 hash key from a byte slice at `bytes[start..start+word_len]`.
/// Uses unaligned u64 read when possible for speed, with length XOR for >7 bytes.
#[inline(always)]
unsafe fn word_hash_key(bytes: &[u8], start: usize, word_len: usize, buf_len: usize) -> u64 {
    if word_len <= 7 {
        if start + 8 <= buf_len {
            let raw = std::ptr::read_unaligned(bytes.as_ptr().add(start) as *const u64);
            raw & ((1u64 << (word_len * 8)) - 1)
        } else {
            let mut k: u64 = 0;
            for (j, &wb) in bytes[start..start + word_len].iter().enumerate() {
                k |= (wb as u64) << (j * 8);
            }
            k
        }
    } else {
        let raw = if start + 8 <= buf_len {
            std::ptr::read_unaligned(bytes.as_ptr().add(start) as *const u64)
        } else {
            let mut k: u64 = 0;
            for (j, &wb) in bytes[start..start + 7.min(buf_len - start)]
                .iter()
                .enumerate()
            {
                k |= (wb as u64) << (j * 8);
            }
            k
        };
        raw ^ ((word_len as u64) << 56)
    }
}

/// Detect repeating period in a byte slice using SIMD-accelerated memchr.
/// Returns the period P if bytes[0..P] == bytes[P..2P], else 0.
#[inline]
unsafe fn detect_text_period(bytes: &[u8], len: usize) -> usize {
    if len < 4 {
        return 0;
    }
    let first_byte = *bytes.get_unchecked(0);
    let max_search = len.min(1025);
    let mut search_from = 0usize;
    while search_from + 1 < max_search {
        match memchr::memchr(first_byte, &bytes[search_from + 1..max_search]) {
            Some(offset) => {
                let p = search_from + offset + 1;
                if len >= p * 2 && bytes[..p] == bytes[p..p * 2] {
                    return p;
                }
                search_from = p;
            }
            None => break,
        }
    }
    0
}

/// Branchless word count: scan bytes and count word starts using init/body tables.
#[inline(always)]
unsafe fn count_words_branchless(
    bytes: &[u8],
    start: usize,
    end: usize,
    is_init: &[u8; 256],
    is_body: &[u8; 256],
) -> usize {
    let mut count = 0usize;
    let mut in_word = 0u8;
    for i in start..end {
        let b = *bytes.get_unchecked(i);
        let cur_init = *is_init.get_unchecked(b as usize);
        let cur_body = *is_body.get_unchecked(b as usize);
        let starts = cur_init & !in_word;
        count += starts as usize;
        in_word = cur_body & (starts | in_word);
    }
    count
}

/// Fill a pointer array using memcpy doubling from a seed of `seed_len` elements
/// already written at the start of `ob_item`, up to `total_len` elements.
#[inline(always)]
unsafe fn memcpy_double_fill(
    ob_item: *mut *mut pyo3::ffi::PyObject,
    seed_len: usize,
    total_len: usize,
) {
    let mut filled = seed_len;
    while filled < total_len {
        let copy_len = (total_len - filled).min(filled);
        std::ptr::copy_nonoverlapping(ob_item, ob_item.add(filled), copy_len);
        filled += copy_len;
    }
}

/// Detect cyclic pattern in a PyList. Returns the period if found, 0 otherwise.
/// A period P means items[0..P] repeats throughout the list.
#[inline]
unsafe fn detect_list_cycle(
    list_ptr: *mut pyo3::ffi::PyObject,
    n: pyo3::ffi::Py_ssize_t,
) -> pyo3::ffi::Py_ssize_t {
    if n <= 1 {
        return 0;
    }
    let first = pyo3::ffi::PyList_GET_ITEM(list_ptr, 0);
    let detect_limit = n.min(128);
    let mut period: pyo3::ffi::Py_ssize_t = 0;
    for i in 1..detect_limit {
        if pyo3::ffi::PyList_GET_ITEM(list_ptr, i) == first {
            period = i;
            break;
        }
    }
    if period > 0 && n >= period * 2 {
        for i in 0..period {
            if pyo3::ffi::PyList_GET_ITEM(list_ptr, i)
                != pyo3::ffi::PyList_GET_ITEM(list_ptr, period + i)
            {
                return 0;
            }
        }
        period
    } else {
        0
    }
}

/// Hash-based pointer cache for batch counting.
/// Caches match results per unique Python object pointer, avoids re-parsing
/// identical strings. `test_fn` receives a raw `PyObject*` and returns whether
/// it matches.
#[inline]
unsafe fn hash_cache_batch_count(
    in_ptr: *mut pyo3::ffi::PyObject,
    n: pyo3::ffi::Py_ssize_t,
    test_fn: impl Fn(*mut pyo3::ffi::PyObject) -> bool,
) -> usize {
    const HASH_BITS: usize = 5;
    const HASH_SIZE: usize = 1 << HASH_BITS;
    const HASH_MASK: usize = HASH_SIZE - 1;
    let mut count = 0;
    let mut hash_ptrs: [*mut pyo3::ffi::PyObject; HASH_SIZE] = [std::ptr::null_mut(); HASH_SIZE];
    let mut hash_vals: [u8; HASH_SIZE] = [0; HASH_SIZE];

    let mut filled = 0usize;
    for i in 0..n {
        let item = pyo3::ffi::PyList_GET_ITEM(in_ptr, i);
        let hash = (item as usize >> 4) ^ (item as usize >> 12);
        let mut slot = hash & HASH_MASK;
        let matched;
        let mut probes = 0usize;
        loop {
            let cached_ptr = *hash_ptrs.get_unchecked(slot);
            if cached_ptr == item {
                matched = *hash_vals.get_unchecked(slot) == 1;
                break;
            }
            if cached_ptr.is_null() && filled < HASH_SIZE - 1 {
                let result = test_fn(item);
                *hash_ptrs.get_unchecked_mut(slot) = item;
                *hash_vals.get_unchecked_mut(slot) = if result { 1 } else { 2 };
                filled += 1;
                matched = result;
                break;
            }
            probes += 1;
            if probes >= HASH_SIZE {
                // Table full, bypass cache
                matched = test_fn(item);
                break;
            }
            slot = (slot + 1) & HASH_MASK;
        }
        if matched {
            count += 1;
        }
    }
    count
}

/// Build a PyList from u8 token indices into a unique_tokens vec.
/// Uses bulk INCREF for efficient reference counting.
#[inline]
unsafe fn build_indexed_pylist<'py>(
    py: Python<'py>,
    indices: &[u8],
    unique_tokens: &[Bound<'py, PyString>],
) -> PyResult<Bound<'py, PyList>> {
    let out_n = indices.len() as pyo3::ffi::Py_ssize_t;
    let list_ptr = pyo3::ffi::PyList_New(out_n);
    if list_ptr.is_null() {
        return Err(pyo3::PyErr::fetch(py));
    }
    let num_unique = unique_tokens.len();
    let mut counts = [0u32; 256];
    for (j, &idx) in indices.iter().enumerate() {
        *counts.get_unchecked_mut(idx as usize) += 1;
        let item_ptr = unique_tokens.get_unchecked(idx as usize).as_ptr();
        pyo3::ffi::PyList_SET_ITEM(list_ptr, j as pyo3::ffi::Py_ssize_t, item_ptr);
    }
    for i in 0..num_unique {
        let ptr = unique_tokens.get_unchecked(i).as_ptr();
        let c = *counts.get_unchecked(i);
        if c > 0 {
            bulk_incref(ptr, c as usize);
        }
    }
    Ok(Bound::from_owned_ptr(py, list_ptr).downcast_into_unchecked())
}

// ============================================================================
// Generic batch/search helpers for any ParserElement
// ============================================================================

/// Generic search_string_count: count matches by scanning with try_match_at
fn generic_search_string_count(parser: &dyn ParserElement, s: &str) -> usize {
    let mut count = 0;
    let mut loc = 0;
    while loc < s.len() {
        if let Some(end) = parser.try_match_at(s, loc) {
            count += 1;
            loc = if end > loc { end } else { loc + 1 };
        } else {
            loc += 1;
        }
    }
    count
}

/// Generic search_string: two-pass with raw FFI for zero PyO3 overhead.
/// Pass 1: count matches. Pass 2: allocate exact-size list and fill.
fn generic_search_string<'py>(
    py: Python<'py>,
    parser: &dyn ParserElement,
    s: &str,
) -> PyResult<Bound<'py, PyList>> {
    unsafe {
        // Pass 1: collect match positions
        let mut matches: Vec<(usize, usize)> = Vec::new();
        let mut loc = 0;
        while loc < s.len() {
            if let Some(end) = parser.try_match_at(s, loc) {
                if end > loc {
                    matches.push((loc, end));
                }
                loc = if end > loc { end } else { loc + 1 };
            } else {
                loc += 1;
            }
        }

        let n = matches.len() as pyo3::ffi::Py_ssize_t;
        if n == 0 {
            return Ok(PyList::empty(py));
        }

        // Pass 2: build list with raw FFI + dedup
        let list_ptr = pyo3::ffi::PyList_New(n);
        if list_ptr.is_null() {
            return Err(pyo3::PyErr::fetch(py));
        }

        let mut dedup: FxHashMap<&str, *mut pyo3::ffi::PyObject> = FxHashMap::default();
        for (i, &(start, end)) in matches.iter().enumerate() {
            let matched = &s[start..end];
            let py_str = if let Some(&existing) = dedup.get(matched) {
                pyo3::ffi::Py_INCREF(existing);
                existing
            } else {
                let new_str = PyString::new(py, matched).into_ptr();
                dedup.insert(matched, new_str);
                pyo3::ffi::Py_INCREF(new_str);
                new_str
            };
            pyo3::ffi::PyList_SET_ITEM(list_ptr, i as pyo3::ffi::Py_ssize_t, py_str);
        }

        // Drop extra refs from dedup (each entry has one extra from creation)
        for (_, ptr) in dedup {
            pyo3::ffi::Py_DECREF(ptr);
        }

        Ok(Bound::from_owned_ptr(py, list_ptr).downcast_into_unchecked())
    }
}

/// Generic parse_string: parse and return results as a PyList of PyStrings.
/// Uses parse_string (full parse) to preserve multi-token results for
/// repetition combinators like ZeroOrMore and OneOrMore.
fn generic_parse_string<'py>(
    py: Python<'py>,
    parser: &dyn ParserElement,
    s: &str,
) -> PyResult<Bound<'py, PyList>> {
    match parser.parse_string(s) {
        Ok(results) => {
            let tokens = results.as_vec();
            unsafe {
                let n = tokens.len() as pyo3::ffi::Py_ssize_t;
                let list_ptr = pyo3::ffi::PyList_New(n);
                if list_ptr.is_null() {
                    return Err(pyo3::PyErr::fetch(py));
                }
                for (i, tok) in tokens.iter().enumerate() {
                    pyo3::ffi::PyList_SET_ITEM(
                        list_ptr,
                        i as pyo3::ffi::Py_ssize_t,
                        PyString::new(py, tok).into_ptr(),
                    );
                }
                Ok(Bound::from_owned_ptr(py, list_ptr).downcast_into_unchecked())
            }
        }
        Err(e) => Err(PyValueError::new_err(e.to_string())),
    }
}

/// Generic parse_batch_count: uniform + cycle + hash cache for dedup
fn generic_parse_batch_count(
    parser: &dyn ParserElement,
    inputs: &Bound<'_, PyList>,
) -> PyResult<usize> {
    unsafe {
        let in_ptr = inputs.as_ptr();
        let n = pyo3::ffi::PyList_GET_SIZE(in_ptr);
        if n == 0 {
            return Ok(0);
        }
        // Uniform path: all items same → check once, multiply
        if list_all_same(in_ptr, n) {
            let item = pyo3::ffi::PyList_GET_ITEM(in_ptr, 0);
            let s = py_str_as_str(item);
            return Ok(if parser.try_match_at(s, 0).is_some() {
                n as usize
            } else {
                0
            });
        }
        // Cycle detection
        let period = detect_list_cycle(in_ptr, n);
        if period > 0 {
            let mut cycle_count = 0usize;
            for i in 0..period {
                let item = pyo3::ffi::PyList_GET_ITEM(in_ptr, i);
                let s = py_str_as_str(item);
                if parser.try_match_at(s, 0).is_some() {
                    cycle_count += 1;
                }
            }
            let num_cycles = n / period;
            let rem = n % period;
            let mut total = cycle_count * num_cycles as usize;
            for i in 0..rem {
                let item = pyo3::ffi::PyList_GET_ITEM(in_ptr, num_cycles * period + i);
                let s = py_str_as_str(item);
                if parser.try_match_at(s, 0).is_some() {
                    total += 1;
                }
            }
            return Ok(total);
        }
        // Hash-based pointer cache
        Ok(hash_cache_batch_count(in_ptr, n, |item| {
            let s = py_str_as_str(item);
            parser.try_match_at(s, 0).is_some()
        }))
    }
}

/// Generic parse_batch: parse each input and return list of result lists.
/// Uses parse_impl to preserve multi-token results for repetition combinators.
fn generic_parse_batch<'py>(
    py: Python<'py>,
    parser: &dyn ParserElement,
    inputs: &Bound<'py, PyList>,
) -> PyResult<Bound<'py, PyList>> {
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

        // Uniform path: all items are the same Python string object
        if list_all_same(in_ptr, n) {
            let item = pyo3::ffi::PyList_GET_ITEM(in_ptr, 0);
            let s = py_str_as_str(item);
            let mut ctx = crate::core::context::ParseContext::new(s);
            let template = match parser.parse_impl(&mut ctx, 0) {
                Ok((_end, results)) => {
                    let tokens = results.as_vec();
                    let inner = pyo3::ffi::PyList_New(tokens.len() as pyo3::ffi::Py_ssize_t);
                    for (j, tok) in tokens.iter().enumerate() {
                        let py_str = PyString::new(py, tok);
                        pyo3::ffi::PyList_SET_ITEM(
                            inner,
                            j as pyo3::ffi::Py_ssize_t,
                            py_str.into_ptr(),
                        );
                    }
                    inner
                }
                Err(_) => pyo3::ffi::PyList_New(0),
            };
            // Fill all slots using PySequence_Repeat (C-level INCREF)
            pyo3::ffi::Py_DECREF(out_ptr); // drop pre-allocated list
            pyo3::ffi::Py_INCREF(template);
            let wrapper = pyo3::ffi::PyList_New(1);
            pyo3::ffi::PyList_SET_ITEM(wrapper, 0, template);
            let result = pyo3::ffi::PySequence_Repeat(wrapper, n);
            pyo3::ffi::Py_DECREF(wrapper);
            if result.is_null() {
                return Err(pyo3::PyErr::fetch(py));
            }
            return Ok(Bound::from_owned_ptr(py, result).downcast_into_unchecked());
        } else {
            // Mixed path: parse each input individually
            for i in 0..n {
                let item = pyo3::ffi::PyList_GET_ITEM(in_ptr, i);
                let s = py_str_as_str(item);
                let mut ctx = crate::core::context::ParseContext::new(s);
                let inner_list = match parser.parse_impl(&mut ctx, 0) {
                    Ok((_end, results)) => {
                        let tokens = results.as_vec();
                        let inner = pyo3::ffi::PyList_New(tokens.len() as pyo3::ffi::Py_ssize_t);
                        for (j, tok) in tokens.iter().enumerate() {
                            let py_str = PyString::new(py, tok);
                            pyo3::ffi::PyList_SET_ITEM(
                                inner,
                                j as pyo3::ffi::Py_ssize_t,
                                py_str.into_ptr(),
                            );
                        }
                        inner
                    }
                    Err(_) => pyo3::ffi::PyList_New(0),
                };
                pyo3::ffi::PyList_SET_ITEM(out_ptr, i, inner_list);
            }
        }

        Ok(Bound::from_owned_ptr(py, out_ptr).downcast_into_unchecked())
    }
}

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
struct PyKeyword {
    inner: Arc<RustKeyword>,
    cached_pystr: Py<PyString>,
}

impl Clone for PyKeyword {
    fn clone(&self) -> Self {
        Python::with_gil(|py| Self {
            inner: self.inner.clone(),
            cached_pystr: self.cached_pystr.clone_ref(py),
        })
    }
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
    // If `other` is already a MatchFirst, flatten its elements
    if let Ok(mf) = other.extract::<PyMatchFirst>() {
        let mut elements = vec![a];
        elements.extend(mf.inner.elements().iter().cloned());
        Ok(PyMatchFirst {
            inner: Arc::new(RustMatchFirst::new(elements)),
        })
    } else {
        let b = extract_parser(other)
            .map_err(|_| PyValueError::new_err("Unsupported operand type for |"))?;
        Ok(PyMatchFirst {
            inner: Arc::new(RustMatchFirst::new(vec![a, b])),
        })
    }
}

/// Like make_or, but called from PyMatchFirst::__or__ where `self` is already a MatchFirst.
/// Flattens both sides.
fn make_or_from_matchfirst(
    existing: &Arc<RustMatchFirst>,
    other: &Bound<'_, PyAny>,
) -> PyResult<PyMatchFirst> {
    let mut elements: Vec<Arc<dyn ParserElement>> = existing.elements().to_vec();
    if let Ok(mf) = other.extract::<PyMatchFirst>() {
        elements.extend(mf.inner.elements().iter().cloned());
    } else {
        let b = extract_parser(other)
            .map_err(|_| PyValueError::new_err("Unsupported operand type for |"))?;
        elements.push(b);
    }
    Ok(PyMatchFirst {
        inner: Arc::new(RustMatchFirst::new(elements)),
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
            let all_same = list_all_same(in_ptr, n);

            if all_same {
                // Fast uniform path: parse once, use PySequence_Repeat for C-level INCREF
                let s = py_str_as_bytes(first_item);
                let matched =
                    s.len() >= match_len && s[0] == first && s[..match_len] == *match_bytes;
                let inner = if matched { matched_ptr } else { empty_ptr };

                pyo3::ffi::Py_DECREF(out_ptr); // drop pre-allocated list, use repeat instead
                pyo3::ffi::Py_INCREF(inner);
                let template = pyo3::ffi::PyList_New(1);
                pyo3::ffi::PyList_SET_ITEM(template, 0, inner);
                let result = pyo3::ffi::PySequence_Repeat(template, n);
                pyo3::ffi::Py_DECREF(template);
                if result.is_null() {
                    return Err(pyo3::PyErr::fetch(py));
                }
                return Ok(Bound::from_owned_ptr(py, result).downcast_into_unchecked());
            } else {
                // Mixed path: last-pointer cache
                let mut last_item: *mut pyo3::ffi::PyObject = std::ptr::null_mut();
                let mut last_matched = false;

                for i in 0..n {
                    let item = pyo3::ffi::PyList_GET_ITEM(in_ptr, i);

                    let matched = if item == last_item {
                        last_matched
                    } else {
                        let s = py_str_as_bytes(item);
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
        if match_len > 0 {
            unsafe {
                let period = detect_text_period(bytes, len);
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
            let all_same = list_all_same(in_ptr, n);

            if all_same {
                // Parse once, return n or 0
                let s = py_str_as_bytes(first_item);
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
                    let s = py_str_as_bytes(item);
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

            let period = detect_list_cycle(in_ptr, n);

            if period > 0 {
                let p = period;
                let mut unique_tokens: Vec<Bound<'py, PyString>> = Vec::new();
                let mut cycle_indices: Vec<u8> = Vec::with_capacity(p as usize);

                // Parse only the first cycle
                for i in 0..p {
                    let item = pyo3::ffi::PyList_GET_ITEM(in_ptr, i);
                    let s_bytes = py_str_as_bytes(item);

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
                        bulk_incref(ptr, c as usize);
                    }
                }
                let list_ptr = pyo3::ffi::PyList_New(total_out as pyo3::ffi::Py_ssize_t);
                if list_ptr.is_null() {
                    return Err(pyo3::PyErr::fetch(py));
                }
                let ob_item = list_ob_item(list_ptr);
                for (j, &ptr) in cycle_ptrs.iter().enumerate() {
                    *ob_item.add(j) = ptr;
                }
                let full_cycles_items = mpc * num_cycles as usize;
                memcpy_double_fill(ob_item, mpc, full_cycles_items);
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

            // --- Fallback: direct output using FxHashMap dedup ---
            // Handles unlimited unique strings safely.
            let mut items: Vec<*mut pyo3::ffi::PyObject> = Vec::with_capacity(n as usize);
            let mut dedup: FxHashMap<*mut pyo3::ffi::PyObject, *mut pyo3::ffi::PyObject> =
                FxHashMap::default();
            let mut _keep_alive: Vec<Bound<'py, PyString>> = Vec::new();

            for i in 0..n {
                let item = pyo3::ffi::PyList_GET_ITEM(in_ptr, i);

                let ptr = match dedup.get(&item) {
                    Some(&cached) => {
                        if !cached.is_null() {
                            pyo3::ffi::Py_INCREF(cached);
                            items.push(cached);
                        }
                        continue;
                    }
                    None => {
                        let s_bytes = py_str_as_bytes(item);
                        if s_bytes.is_empty() || !self.inner.init_chars_contains(s_bytes[0]) {
                            dedup.insert(item, std::ptr::null_mut());
                            continue;
                        }
                        let mut end = 1;
                        while end < s_bytes.len() && self.inner.body_chars_contains(s_bytes[end]) {
                            end += 1;
                        }
                        let py_str = if end == s_bytes.len() {
                            pyo3::ffi::Py_INCREF(item);
                            Bound::from_owned_ptr(py, item).downcast_into_unchecked()
                        } else {
                            let s = std::str::from_utf8_unchecked(s_bytes);
                            PyString::new(py, &s[..end])
                        };
                        let p = py_str.as_ptr();
                        pyo3::ffi::Py_INCREF(p); // for the output list slot
                        dedup.insert(item, p);
                        _keep_alive.push(py_str);
                        p
                    }
                };
                items.push(ptr);
            }

            let out_n = items.len() as pyo3::ffi::Py_ssize_t;
            let list_ptr = pyo3::ffi::PyList_New(out_n);
            if list_ptr.is_null() {
                return Err(pyo3::PyErr::fetch(py));
            }
            for (j, &ptr) in items.iter().enumerate() {
                pyo3::ffi::PyList_SET_ITEM(list_ptr, j as pyo3::ffi::Py_ssize_t, ptr);
            }
            Ok(Bound::from_owned_ptr(py, list_ptr).downcast_into_unchecked())
        }
    }

    /// Count word matches in batch — hash-based pointer cache
    fn parse_batch_count(&self, inputs: &Bound<'_, PyList>) -> PyResult<usize> {
        unsafe {
            let in_ptr = inputs.as_ptr();
            let n = pyo3::ffi::PyList_GET_SIZE(in_ptr);
            Ok(hash_cache_batch_count(in_ptr, n, |item| {
                let s_bytes = py_str_as_bytes(item);
                !s_bytes.is_empty() && self.inner.init_chars_contains(s_bytes[0])
            }))
        }
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
            let period = detect_text_period(bytes, len);
            if period > 0 {
                let first_byte = *bytes.get_unchecked(0);
                // Verify no word spans the cycle boundary:
                // The last byte of cycle must NOT be a body char, OR the first byte of next cycle must NOT be init/body
                let last_of_cycle = *bytes.get_unchecked(period - 1);
                let last_is_body = *is_body.get_unchecked(last_of_cycle as usize);
                let first_is_init = *is_init.get_unchecked(first_byte as usize);
                let first_is_body = *is_body.get_unchecked(first_byte as usize);
                let word_spans_boundary =
                    last_is_body != 0 && (first_is_init != 0 || first_is_body != 0);

                if !word_spans_boundary {
                    let cycle_count = count_words_branchless(bytes, 0, period, &is_init, &is_body);
                    let full_cycles = len / period;
                    let rem_start = full_cycles * period;
                    let total = cycle_count * full_cycles
                        + count_words_branchless(bytes, rem_start, len, &is_init, &is_body);
                    return total;
                }
            }
        }

        // Fallback: full branchless scan
        unsafe { count_words_branchless(bytes, 0, len, &is_init, &is_body) }
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
        let mut is_init = [0u8; 256];
        let mut is_body = [0u8; 256];
        for b in 0u16..256 {
            is_init[b as usize] = self.inner.init_chars_contains(b as u8) as u8;
            is_body[b as usize] = self.inner.body_chars_contains(b as u8) as u8;
        }

        unsafe {
            // --- Text repetition fast path ---
            'cycle: {
                let period = detect_text_period(bytes, len);
                if period == 0 {
                    break 'cycle;
                }

                // Scan first cycle to find word boundaries
                let mut cycle_word_ranges: Vec<(usize, usize)> = Vec::new();
                let mut pos = 0usize;
                while pos < period {
                    let b = *bytes.get_unchecked(pos);
                    if is_init[b as usize] == 0 {
                        pos += 1;
                        continue;
                    }
                    let start = pos;
                    pos += 1;
                    while pos < period && is_body[*bytes.get_unchecked(pos) as usize] != 0 {
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
                    && is_body[*bytes.get_unchecked(period) as usize] != 0
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
                    let key = word_hash_key(bytes, start, word_len, len);

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

                // Fast path: no remainder — use PySequence_Repeat (C-level INCREF)
                if remainder == 0 && wpc > 0 {
                    let cycle_list = pyo3::ffi::PyList_New(wpc as pyo3::ffi::Py_ssize_t);
                    if cycle_list.is_null() {
                        return Err(pyo3::PyErr::fetch(py));
                    }
                    for (j, &ptr) in cycle_ptrs.iter().enumerate() {
                        pyo3::ffi::Py_INCREF(ptr);
                        pyo3::ffi::PyList_SET_ITEM(cycle_list, j as pyo3::ffi::Py_ssize_t, ptr);
                    }
                    let result = pyo3::ffi::PySequence_Repeat(
                        cycle_list,
                        num_full_cycles as pyo3::ffi::Py_ssize_t,
                    );
                    pyo3::ffi::Py_DECREF(cycle_list);
                    if result.is_null() {
                        return Err(pyo3::PyErr::fetch(py));
                    }
                    return Ok(Bound::from_owned_ptr(py, result).downcast_into_unchecked());
                }

                // Scan remainder for words (partial last cycle)
                let rem_start_byte = num_full_cycles * period;
                let mut rem_ptrs: Vec<*mut pyo3::ffi::PyObject> = Vec::new();
                if remainder > 0 {
                    let mut rpos = rem_start_byte;
                    while rpos < len {
                        let b = *bytes.get_unchecked(rpos);
                        if is_init[b as usize] == 0 {
                            rpos += 1;
                            continue;
                        }
                        let wstart = rpos;
                        rpos += 1;
                        while rpos < len && is_body[*bytes.get_unchecked(rpos) as usize] != 0 {
                            rpos += 1;
                        }
                        let wlen = rpos - wstart;
                        let key = word_hash_key(bytes, wstart, wlen, len);

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

                    let ob_item = list_ob_item(list_ptr);

                    if full_items > wpc {
                        memcpy_double_fill(ob_item, wpc, full_items);
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
                        bulk_incref(sptr, count);
                    }
                }

                return Ok(Bound::from_owned_ptr(py, list_ptr).downcast_into_unchecked());
            }

            // --- Fallback: full-scan approach (non-cyclic text) ---
            // Uses FxHashMap for dedup (handles unlimited unique words).
            // Two-pass: count first, then build list directly.
            let word_count = count_words_branchless(bytes, 0, len, &is_init, &is_body);
            if word_count == 0 {
                return Ok(PyList::empty(py));
            }

            let list_ptr = pyo3::ffi::PyList_New(word_count as pyo3::ffi::Py_ssize_t);
            if list_ptr.is_null() {
                return Err(pyo3::PyErr::fetch(py));
            }

            let mut dedup: FxHashMap<&str, *mut pyo3::ffi::PyObject> = FxHashMap::default();
            let mut _keep_alive: Vec<Bound<'py, PyString>> = Vec::new();
            let mut out_idx = 0usize;

            let mut pos = 0;
            while pos < len {
                let b = *bytes.get_unchecked(pos);
                if is_init[b as usize] == 0 {
                    pos += 1;
                    continue;
                }
                let start = pos;
                pos += 1;
                while pos < len {
                    let b2 = *bytes.get_unchecked(pos);
                    if is_body[b2 as usize] == 0 {
                        break;
                    }
                    pos += 1;
                }

                let word = std::str::from_utf8_unchecked(&bytes[start..pos]);
                let ptr = match dedup.get(word) {
                    Some(&p) => {
                        pyo3::ffi::Py_INCREF(p);
                        p
                    }
                    None => {
                        let py_str = PyString::new(py, word);
                        let p = py_str.as_ptr();
                        pyo3::ffi::Py_INCREF(p); // one for the list slot
                        dedup.insert(word, p);
                        _keep_alive.push(py_str);
                        p
                    }
                };
                pyo3::ffi::PyList_SET_ITEM(list_ptr, out_idx as pyo3::ffi::Py_ssize_t, ptr);
                out_idx += 1;
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

            let period = detect_list_cycle(in_ptr, n);

            if period > 0 {
                let p = period;
                let mut unique_tokens: Vec<Bound<'py, PyString>> = Vec::new();
                let mut cycle_indices: Vec<u8> = Vec::with_capacity(p as usize);

                for i in 0..p {
                    let item = pyo3::ffi::PyList_GET_ITEM(in_ptr, i);
                    let s = py_str_as_str(item);

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
                        bulk_incref(ptr, c as usize);
                    }
                }
                let list_ptr = pyo3::ffi::PyList_New(total_out as pyo3::ffi::Py_ssize_t);
                if list_ptr.is_null() {
                    return Err(pyo3::PyErr::fetch(py));
                }
                let ob_item = list_ob_item(list_ptr);
                for (j, &ptr) in cycle_ptrs.iter().enumerate() {
                    *ob_item.add(j) = ptr;
                }
                let full_cycles_items = mpc * num_cycles as usize;
                memcpy_double_fill(ob_item, mpc, full_cycles_items);
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

            // --- Fallback: FxHashMap-based dedup (handles unlimited unique strings) ---
            let mut items: Vec<*mut pyo3::ffi::PyObject> = Vec::with_capacity(n as usize);
            let mut dedup: FxHashMap<*mut pyo3::ffi::PyObject, *mut pyo3::ffi::PyObject> =
                FxHashMap::default();
            let mut _keep_alive: Vec<Bound<'py, PyString>> = Vec::new();

            for i in 0..n {
                let item = pyo3::ffi::PyList_GET_ITEM(in_ptr, i);

                match dedup.get(&item) {
                    Some(&cached) => {
                        if !cached.is_null() {
                            pyo3::ffi::Py_INCREF(cached);
                            items.push(cached);
                        }
                        continue;
                    }
                    None => {
                        let s = py_str_as_str(item);
                        match self.inner.try_match(s) {
                            Some(matched) => {
                                let py_str = if matched.len() == s.len() {
                                    pyo3::ffi::Py_INCREF(item);
                                    Bound::from_owned_ptr(py, item).downcast_into_unchecked()
                                } else {
                                    PyString::new(py, matched)
                                };
                                let p = py_str.as_ptr();
                                pyo3::ffi::Py_INCREF(p);
                                dedup.insert(item, p);
                                _keep_alive.push(py_str);
                                items.push(p);
                            }
                            None => {
                                dedup.insert(item, std::ptr::null_mut());
                            }
                        }
                    }
                };
            }

            let out_n = items.len() as pyo3::ffi::Py_ssize_t;
            let list_ptr = pyo3::ffi::PyList_New(out_n);
            if list_ptr.is_null() {
                return Err(pyo3::PyErr::fetch(py));
            }
            for (j, &ptr) in items.iter().enumerate() {
                pyo3::ffi::PyList_SET_ITEM(list_ptr, j as pyo3::ffi::Py_ssize_t, ptr);
            }
            Ok(Bound::from_owned_ptr(py, list_ptr).downcast_into_unchecked())
        }
    }

    /// Count regex matches in batch — hash-based pointer cache
    fn parse_batch_count(&self, inputs: &Bound<'_, PyList>) -> PyResult<usize> {
        unsafe {
            let in_ptr = inputs.as_ptr();
            let n = pyo3::ffi::PyList_GET_SIZE(in_ptr);
            Ok(hash_cache_batch_count(in_ptr, n, |item| {
                let s = py_str_as_str(item);
                self.inner.try_match(s).is_some()
            }))
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
impl PyKeyword {
    #[new]
    fn new(py: Python<'_>, s: &str) -> Self {
        Self {
            inner: Arc::new(RustKeyword::new(s)),
            cached_pystr: PyString::new(py, s).unbind(),
        }
    }

    /// Fast keyword parse — uses try_match_at + cached PyString, zero allocation
    fn parse_string<'py>(&self, py: Python<'py>, s: &str) -> PyResult<Bound<'py, PyList>> {
        match self.inner.try_match_at(s, 0) {
            Some(_end) => PyList::new(py, [self.cached_pystr.bind(py)]),
            None => Err(PyValueError::new_err("Expected keyword")),
        }
    }

    fn matches(&self, s: &str) -> bool {
        self.inner.try_match_at(s, 0).is_some()
    }

    fn search_string_count(&self, s: &str) -> usize {
        generic_search_string_count(self.inner.as_ref(), s)
    }

    /// Search string — count + PySequence_Repeat (same pattern as Literal)
    fn search_string<'py>(&self, py: Python<'py>, s: &str) -> PyResult<Bound<'py, PyList>> {
        let cached = self.cached_pystr.bind(py);
        let count = self.search_string_count(s);
        if count == 0 {
            return Ok(PyList::empty(py));
        }
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

    fn parse_batch_count(&self, inputs: &Bound<'_, PyList>) -> PyResult<usize> {
        unsafe {
            let in_ptr = inputs.as_ptr();
            let n = pyo3::ffi::PyList_GET_SIZE(in_ptr);
            if n == 0 {
                return Ok(0);
            }
            if list_all_same(in_ptr, n) {
                let item = pyo3::ffi::PyList_GET_ITEM(in_ptr, 0);
                let s = py_str_as_str(item);
                return Ok(if self.inner.try_match_at(s, 0).is_some() {
                    n as usize
                } else {
                    0
                });
            }
            Ok(hash_cache_batch_count(in_ptr, n, |item| {
                let s = py_str_as_str(item);
                self.inner.try_match_at(s, 0).is_some()
            }))
        }
    }

    /// Specialized parse_batch — cached PyString + last-pointer cache
    fn parse_batch<'py>(
        &self,
        py: Python<'py>,
        inputs: &Bound<'py, PyList>,
    ) -> PyResult<Bound<'py, PyList>> {
        let cached = self.cached_pystr.bind(py);
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

            // Uniform path
            if list_all_same(in_ptr, n) {
                let item = pyo3::ffi::PyList_GET_ITEM(in_ptr, 0);
                let s = py_str_as_str(item);
                let inner = if self.inner.try_match_at(s, 0).is_some() {
                    matched_ptr
                } else {
                    empty_ptr
                };
                pyo3::ffi::Py_INCREF(inner);
                let template = pyo3::ffi::PyList_New(1);
                pyo3::ffi::PyList_SET_ITEM(template, 0, inner);
                let result = pyo3::ffi::PySequence_Repeat(template, n);
                pyo3::ffi::Py_DECREF(template);
                if result.is_null() {
                    return Err(pyo3::PyErr::fetch(py));
                }
                return Ok(Bound::from_owned_ptr(py, result).downcast_into_unchecked());
            }

            // Mixed path: last-pointer cache
            let out_ptr = pyo3::ffi::PyList_New(n);
            if out_ptr.is_null() {
                return Err(pyo3::PyErr::fetch(py));
            }
            let mut last_item: *mut pyo3::ffi::PyObject = std::ptr::null_mut();
            let mut last_matched = false;

            for i in 0..n {
                let item = pyo3::ffi::PyList_GET_ITEM(in_ptr, i);
                let matched = if item == last_item {
                    last_matched
                } else {
                    let s = py_str_as_str(item);
                    let result = self.inner.try_match_at(s, 0).is_some();
                    last_item = item;
                    last_matched = result;
                    result
                };
                let inner = if matched { matched_ptr } else { empty_ptr };
                pyo3::ffi::Py_INCREF(inner);
                pyo3::ffi::PyList_SET_ITEM(out_ptr, i, inner);
            }
            Ok(Bound::from_owned_ptr(py, out_ptr).downcast_into_unchecked())
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
impl PyAnd {
    /// Parse using try_match_at on flattened elements — raw FFI list construction
    fn parse_string<'py>(&self, py: Python<'py>, s: &str) -> PyResult<Bound<'py, PyList>> {
        let elements = self.inner.elements();
        unsafe {
            // Collect tokens into a stack buffer (most And sequences are small)
            let mut tokens: Vec<*mut pyo3::ffi::PyObject> = Vec::with_capacity(elements.len());
            let mut pos = 0usize;
            for elem in elements {
                match elem.try_match_at(s, pos) {
                    Some(end) => {
                        let sub = &s[pos..end];
                        if !sub.is_empty() {
                            tokens.push(PyString::new(py, sub).into_ptr());
                        }
                        pos = end;
                    }
                    None => {
                        // Clean up already-created PyStrings
                        for &ptr in &tokens {
                            pyo3::ffi::Py_DECREF(ptr);
                        }
                        return Err(PyValueError::new_err("Expected match"));
                    }
                }
            }
            let n = tokens.len() as pyo3::ffi::Py_ssize_t;
            let list_ptr = pyo3::ffi::PyList_New(n);
            if list_ptr.is_null() {
                for &ptr in &tokens {
                    pyo3::ffi::Py_DECREF(ptr);
                }
                return Err(pyo3::PyErr::fetch(py));
            }
            for (i, &ptr) in tokens.iter().enumerate() {
                pyo3::ffi::PyList_SET_ITEM(list_ptr, i as pyo3::ffi::Py_ssize_t, ptr);
            }
            Ok(Bound::from_owned_ptr(py, list_ptr).downcast_into_unchecked())
        }
    }

    /// Zero-allocation match check using try_match_at (no ParseResults)
    fn matches(&self, s: &str) -> bool {
        self.inner.try_match_at(s, 0).is_some()
    }

    fn search_string_count(&self, s: &str) -> usize {
        generic_search_string_count(self.inner.as_ref(), s)
    }

    /// Search string — raw FFI with dedup for repeated tokens
    fn search_string<'py>(&self, py: Python<'py>, s: &str) -> PyResult<Bound<'py, PyList>> {
        let elements = self.inner.elements();
        unsafe {
            let mut items: Vec<*mut pyo3::ffi::PyObject> = Vec::with_capacity(64);
            let mut dedup: FxHashMap<&str, *mut pyo3::ffi::PyObject> = FxHashMap::default();
            let mut loc = 0;
            while loc < s.len() {
                if let Some(end) = self.inner.try_match_at(s, loc) {
                    let mut pos = loc;
                    for elem in elements {
                        if let Some(elem_end) = elem.try_match_at(s, pos) {
                            let sub = &s[pos..elem_end];
                            if !sub.is_empty() {
                                let py_str = if let Some(&existing) = dedup.get(sub) {
                                    pyo3::ffi::Py_INCREF(existing);
                                    existing
                                } else {
                                    let new_str = PyString::new(py, sub).into_ptr();
                                    dedup.insert(sub, new_str);
                                    pyo3::ffi::Py_INCREF(new_str);
                                    new_str
                                };
                                items.push(py_str);
                            }
                            pos = elem_end;
                        }
                    }
                    loc = if end > loc { end } else { loc + 1 };
                } else {
                    loc += 1;
                }
            }

            let n = items.len() as pyo3::ffi::Py_ssize_t;
            let list_ptr = pyo3::ffi::PyList_New(n);
            if list_ptr.is_null() {
                for &ptr in &items {
                    pyo3::ffi::Py_DECREF(ptr);
                }
                for (_, ptr) in dedup {
                    pyo3::ffi::Py_DECREF(ptr);
                }
                return Err(pyo3::PyErr::fetch(py));
            }
            for (i, &ptr) in items.iter().enumerate() {
                pyo3::ffi::PyList_SET_ITEM(list_ptr, i as pyo3::ffi::Py_ssize_t, ptr);
            }
            // Drop extra refs from dedup
            for (_, ptr) in dedup {
                pyo3::ffi::Py_DECREF(ptr);
            }
            Ok(Bound::from_owned_ptr(py, list_ptr).downcast_into_unchecked())
        }
    }

    /// Cyclic detection + hash-based pointer cache count
    fn parse_batch_count(&self, inputs: &Bound<'_, PyList>) -> PyResult<usize> {
        unsafe {
            let in_ptr = inputs.as_ptr();
            let n = pyo3::ffi::PyList_GET_SIZE(in_ptr);
            if n == 0 {
                return Ok(0);
            }

            let period = detect_list_cycle(in_ptr, n);
            if period > 0 {
                // Count matches in one cycle
                let mut cycle_count = 0usize;
                for i in 0..period {
                    let item = pyo3::ffi::PyList_GET_ITEM(in_ptr, i);
                    let s = py_str_as_str(item);
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
                    let s = py_str_as_str(item);
                    if self.inner.try_match_at(s, 0).is_some() {
                        total += 1;
                    }
                }
                return Ok(total);
            }

            // Fallback: hash-based pointer cache
            Ok(hash_cache_batch_count(in_ptr, n, |item| {
                let s = py_str_as_str(item);
                self.inner.try_match_at(s, 0).is_some()
            }))
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

            let period = detect_list_cycle(in_ptr, n);
            let is_cyclic = period > 0;

            if is_cyclic {
                let p = period;
                let mut unique_tokens: Vec<Bound<'py, PyString>> = Vec::new();
                let mut token_map: FxHashMap<&str, u8> = FxHashMap::default();
                let mut cycle_token_indices: Vec<u8> = Vec::with_capacity(p as usize * elem_count);

                // Parse first cycle
                for i in 0..p {
                    let item = pyo3::ffi::PyList_GET_ITEM(in_ptr, i);
                    let s = py_str_as_str(item);

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
                    let s = py_str_as_str(item);
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
                        bulk_incref(ptr, c as usize);
                    }
                }
                let list_ptr = pyo3::ffi::PyList_New(total_out as pyo3::ffi::Py_ssize_t);
                if list_ptr.is_null() {
                    return Err(pyo3::PyErr::fetch(py));
                }
                let ob_item = list_ob_item(list_ptr);
                for (j, &ptr) in cycle_ptrs.iter().enumerate() {
                    *ob_item.add(j) = ptr;
                }
                let full_cycles_items = tpc * num_cycles as usize;
                memcpy_double_fill(ob_item, tpc, full_cycles_items);
                let mut out_pos = full_cycles_items;
                for &idx in rem_token_indices.iter() {
                    *ob_item.add(out_pos) = unique_tokens.get_unchecked(idx as usize).as_ptr();
                    out_pos += 1;
                }
                return Ok(Bound::from_owned_ptr(py, list_ptr).downcast_into_unchecked());
            }

            // --- Fallback: hash-based approach with probe limit ---
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
            let mut filled = 0usize;

            for i in 0..n {
                let item = pyo3::ffi::PyList_GET_ITEM(in_ptr, i);

                let hash = (item as usize >> 4) ^ (item as usize >> 12);
                let mut slot = hash & HASH_MASK;
                let mut cache_hit = false;
                let mut probes = 0usize;
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
                    if cached_ptr.is_null() && filled < HASH_SIZE - 1 {
                        break;
                    }
                    probes += 1;
                    if probes >= HASH_SIZE {
                        break; // table full, parse without caching
                    }
                    slot = (slot + 1) & HASH_MASK;
                }
                if cache_hit {
                    continue;
                }

                let s = py_str_as_str(item);

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

                // Only cache if slot is available
                if probes < HASH_SIZE {
                    *hash_ptrs.get_unchecked_mut(slot) = item;
                    *hash_matched.get_unchecked_mut(slot) = matched_all;
                    filled += 1;
                    if matched_all {
                        let new_tokens = &token_indices[start_idx..];
                        *hash_start.get_unchecked_mut(slot) = first_tokens.len() as u32;
                        *hash_count.get_unchecked_mut(slot) = new_tokens.len() as u8;
                        first_tokens.extend_from_slice(new_tokens);
                    }
                }
                if !matched_all {
                    token_indices.truncate(start_idx);
                }
            }

            build_indexed_pylist(py, &token_indices, &unique_tokens)
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

    fn parse_string<'py>(&self, py: Python<'py>, s: &str) -> PyResult<Bound<'py, PyList>> {
        let mut ctx = crate::core::context::ParseContext::new(s);
        for elem in self.inner.elements() {
            if let Ok((_end, results)) = elem.parse_impl(&mut ctx, 0) {
                let tokens = results.as_vec();
                unsafe {
                    let n = tokens.len() as pyo3::ffi::Py_ssize_t;
                    let list_ptr = pyo3::ffi::PyList_New(n);
                    if list_ptr.is_null() {
                        return Err(pyo3::PyErr::fetch(py));
                    }
                    for (j, tok) in tokens.iter().enumerate() {
                        pyo3::ffi::PyList_SET_ITEM(
                            list_ptr,
                            j as pyo3::ffi::Py_ssize_t,
                            PyString::new(py, tok).into_ptr(),
                        );
                    }
                    return Ok(Bound::from_owned_ptr(py, list_ptr).downcast_into_unchecked());
                }
            }
        }
        Err(PyValueError::new_err("No match found"))
    }

    fn matches(&self, s: &str) -> bool {
        self.inner.try_match_at(s, 0).is_some()
    }

    fn search_string_count(&self, s: &str) -> usize {
        generic_search_string_count(self.inner.as_ref(), s)
    }

    fn search_string<'py>(&self, py: Python<'py>, s: &str) -> PyResult<Bound<'py, PyList>> {
        generic_search_string(py, self.inner.as_ref(), s)
    }

    fn parse_batch_count(&self, inputs: &Bound<'_, PyList>) -> PyResult<usize> {
        generic_parse_batch_count(self.inner.as_ref(), inputs)
    }

    fn parse_batch<'py>(
        &self,
        py: Python<'py>,
        inputs: &Bound<'py, PyList>,
    ) -> PyResult<Bound<'py, PyList>> {
        generic_parse_batch(py, self.inner.as_ref(), inputs)
    }

    fn __add__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyAnd> {
        make_and(self.inner.clone(), other)
    }

    fn __or__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyMatchFirst> {
        make_or_from_matchfirst(&self.inner, other)
    }
}

/// Generate a complete `#[pymethods]` impl for thin wrapper parser types.
/// These types delegate all methods to generic helpers.
macro_rules! impl_thin_parser_wrapper {
    ($py_type:ident, $rust_type:ident) => {
        #[pymethods]
        impl $py_type {
            #[new]
            fn new(expr: &Bound<'_, PyAny>) -> PyResult<Self> {
                let inner = extract_parser(expr)?;
                Ok(Self {
                    inner: Arc::new($rust_type::new(inner)),
                })
            }
            fn parse_string<'py>(&self, py: Python<'py>, s: &str) -> PyResult<Bound<'py, PyList>> {
                generic_parse_string(py, self.inner.as_ref(), s)
            }
            fn matches(&self, s: &str) -> bool {
                self.inner.try_match_at(s, 0).is_some()
            }
            fn search_string_count(&self, s: &str) -> usize {
                generic_search_string_count(self.inner.as_ref(), s)
            }
            fn search_string<'py>(&self, py: Python<'py>, s: &str) -> PyResult<Bound<'py, PyList>> {
                generic_search_string(py, self.inner.as_ref(), s)
            }
            fn parse_batch_count(&self, inputs: &Bound<'_, PyList>) -> PyResult<usize> {
                generic_parse_batch_count(self.inner.as_ref(), inputs)
            }
            fn parse_batch<'py>(
                &self,
                py: Python<'py>,
                inputs: &Bound<'py, PyList>,
            ) -> PyResult<Bound<'py, PyList>> {
                generic_parse_batch(py, self.inner.as_ref(), inputs)
            }
            fn __add__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyAnd> {
                make_and(self.inner.clone(), other)
            }
            fn __or__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyMatchFirst> {
                make_or(self.inner.clone(), other)
            }
        }
    };
}

impl_thin_parser_wrapper!(PyZeroOrMore, RustZeroOrMore);
impl_thin_parser_wrapper!(PyOneOrMore, RustOneOrMore);
impl_thin_parser_wrapper!(PyOptional, RustOptional);
impl_thin_parser_wrapper!(PyGroup, RustGroup);
impl_thin_parser_wrapper!(PySuppress, RustSuppress);

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

    m.add_function(wrap_pyfunction!(alphas, m)?)?;
    m.add_function(wrap_pyfunction!(alphanums, m)?)?;
    m.add_function(wrap_pyfunction!(nums, m)?)?;
    m.add_function(wrap_pyfunction!(printables, m)?)?;

    m.add("__version__", "0.2.0")?;
    Ok(())
}
