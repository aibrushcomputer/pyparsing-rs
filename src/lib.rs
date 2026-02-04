#![allow(dead_code)]

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyList, PyString};
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
#[derive(Clone)]
struct PyLiteral {
    inner: Arc<RustLiteral>,
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
    let b = extract_parser(other)
        .map_err(|_| PyValueError::new_err("Unsupported operand type for +"))?;
    Ok(PyAnd {
        inner: Arc::new(RustAnd::new(vec![a, b])),
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
    fn new(s: &str) -> Self {
        Self {
            inner: Arc::new(RustLiteral::new(s)),
        }
    }

    /// Fast inline parse — bypasses ParseContext/ParseResults for maximum speed
    fn parse_string(&self, s: &str) -> PyResult<Vec<String>> {
        let match_str = self.inner.match_str();
        let match_bytes = match_str.as_bytes();
        let match_len = match_bytes.len();
        let input_bytes = s.as_bytes();

        if input_bytes.len() >= match_len
            && input_bytes[0] == self.inner.first_byte()
            && input_bytes[..match_len] == *match_bytes
        {
            Ok(vec![match_str.to_string()])
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

    /// Optimized batch parse — inline byte comparison, skip ParseResults
    fn parse_batch(&self, strings: Vec<String>) -> PyResult<Vec<Vec<String>>> {
        let match_bytes = self.inner.match_str().as_bytes();
        let match_len = match_bytes.len();
        let first = self.inner.first_byte();
        let match_str = self.inner.match_str().to_string();

        let results: Vec<Vec<String>> = strings
            .iter()
            .map(|s| {
                let bytes = s.as_bytes();
                if bytes.len() >= match_len
                    && bytes[0] == first
                    && bytes[..match_len] == *match_bytes
                {
                    vec![match_str.clone()]
                } else {
                    Vec::new()
                }
            })
            .collect();
        Ok(results)
    }

    fn search_string(&self, s: &str) -> PyResult<Vec<Vec<String>>> {
        let results = self.inner.search_string(s);
        Ok(results.into_iter().map(|r| r.as_list()).collect())
    }

    /// Count occurrences using SIMD-accelerated memchr — zero Python object allocation
    fn search_string_count(&self, s: &str) -> usize {
        let finder = memchr::memmem::Finder::new(self.inner.match_str());
        let bytes = s.as_bytes();
        let match_len = self.inner.match_str().len();
        let mut count = 0;
        let mut pos = 0;
        while pos < bytes.len() {
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

    /// Count matches in batch — zero-copy string access, no String clones
    fn parse_batch_count(&self, inputs: &Bound<'_, PyList>) -> PyResult<usize> {
        let match_bytes = self.inner.match_str().as_bytes();
        let match_len = match_bytes.len();
        let first = self.inner.first_byte();

        let mut count = 0;
        for item in inputs.iter() {
            let pystr = item.downcast::<PyString>()?;
            let s = pystr.to_str()?;
            let bytes = s.as_bytes();
            if bytes.len() >= match_len && bytes[0] == first && bytes[..match_len] == *match_bytes {
                count += 1;
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

    /// Inlined fast-path word parse — bypasses ParseContext/ParseResults
    fn parse_string(&self, s: &str) -> PyResult<Vec<String>> {
        let bytes = s.as_bytes();
        if bytes.is_empty() || !self.inner.init_chars_contains(bytes[0]) {
            return Err(PyValueError::new_err("Expected word"));
        }
        let mut end = 1;
        while end < bytes.len() && bytes[end] < 128 && self.inner.body_chars_contains(bytes[end]) {
            end += 1;
        }
        // Handle any trailing non-ASCII (rare path)
        while end < bytes.len() && self.inner.body_chars_contains(bytes[end]) {
            end += 1;
        }
        Ok(vec![s[..end].to_string()])
    }

    fn parse_batch(&self, strings: Vec<String>) -> PyResult<Vec<Vec<String>>> {
        let results: Vec<Vec<String>> = strings
            .iter()
            .map(|s| {
                let bytes = s.as_bytes();
                if bytes.is_empty() || !self.inner.init_chars_contains(bytes[0]) {
                    return Vec::new();
                }
                let mut end = 1;
                while end < bytes.len() && self.inner.body_chars_contains(bytes[end]) {
                    end += 1;
                }
                vec![s[..end].to_string()]
            })
            .collect();
        Ok(results)
    }

    /// Count word matches in batch — zero-copy string access, no String clones
    fn parse_batch_count(&self, inputs: &Bound<'_, PyList>) -> PyResult<usize> {
        let mut count = 0;
        for item in inputs.iter() {
            let pystr = item.downcast::<PyString>()?;
            let s = pystr.to_str()?;
            let bytes = s.as_bytes();
            if !bytes.is_empty() && self.inner.init_chars_contains(bytes[0]) {
                count += 1;
            }
        }
        Ok(count)
    }

    /// Count word matches in large text — zero Python object allocation
    fn search_string_count(&self, s: &str) -> usize {
        let mut count = 0;
        let bytes = s.as_bytes();
        let mut pos = 0;
        while pos < bytes.len() {
            // Skip non-init characters
            if !self.inner.init_chars_contains(bytes[pos]) {
                pos += 1;
                continue;
            }
            // Found start of word
            let start = pos;
            pos += 1;
            while pos < bytes.len() && self.inner.body_chars_contains(bytes[pos]) {
                pos += 1;
            }
            if pos > start {
                count += 1;
            }
        }
        count
    }

    /// Zero-allocation match check via try_match_at
    fn matches(&self, s: &str) -> bool {
        self.inner.try_match_at(s, 0).is_some()
    }

    fn search_string(&self, s: &str) -> PyResult<Vec<Vec<String>>> {
        let results = self.inner.search_string(s);
        Ok(results.into_iter().map(|r| r.as_list()).collect())
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

    /// Inlined fast-path regex parse — bypasses ParseContext/ParseResults
    fn parse_string(&self, s: &str) -> PyResult<Vec<String>> {
        match self.inner.try_match(s) {
            Some(matched) => Ok(vec![matched.to_string()]),
            None => Err(PyValueError::new_err("Expected regex match")),
        }
    }

    /// Zero-allocation match check
    fn matches(&self, s: &str) -> bool {
        self.inner.try_match(s).is_some()
    }

    /// Count regex matches in text — zero Python object allocation
    fn search_string_count(&self, s: &str) -> usize {
        let mut count = 0;
        let mut pos = 0;
        while pos < s.len() {
            match self.inner.try_match(&s[pos..]) {
                Some(matched) => {
                    count += 1;
                    pos += matched.len().max(1);
                }
                None => {
                    pos += 1;
                }
            }
        }
        count
    }

    fn search_string(&self, s: &str) -> PyResult<Vec<Vec<String>>> {
        let results = self.inner.search_string(s);
        Ok(results.into_iter().map(|r| r.as_list()).collect())
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
    fn parse_string(&self, s: &str) -> PyResult<Vec<String>> {
        self.inner
            .parse_string(s)
            .map(|r| r.as_list())
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Zero-allocation match check using try_match_at (no ParseResults)
    fn matches(&self, s: &str) -> bool {
        self.inner.try_match_at(s, 0).is_some()
    }

    /// Count matches in batch — zero-copy + zero-alloc via try_match_at
    fn parse_batch_count(&self, inputs: &Bound<'_, PyList>) -> PyResult<usize> {
        let mut count = 0;
        for item in inputs.iter() {
            let pystr = item.downcast::<PyString>()?;
            let s = pystr.to_str()?;
            if self.inner.try_match_at(s, 0).is_some() {
                count += 1;
            }
        }
        Ok(count)
    }

    fn __add__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyAnd> {
        make_and(self.inner.clone(), other)
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

    m.add("__version__", "0.1.0")?;
    Ok(())
}
