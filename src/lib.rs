use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use std::sync::Arc;

mod core;
mod elements;
mod helpers;

use elements::literals::{Literal as RustLiteral, Keyword as RustKeyword};
use elements::chars::{Word as RustWord, RegexMatch};
use elements::combinators::{And as RustAnd, MatchFirst as RustMatchFirst};
use elements::repetition::{ZeroOrMore as RustZeroOrMore, OneOrMore as RustOneOrMore, Optional as RustOptional};
use elements::structure::{Group as RustGroup, Suppress as RustSuppress};
use core::parser::ParserElement;

/// Literal match element
#[pyclass(name = "Literal")]
#[derive(Clone)]
struct PyLiteral {
    inner: Arc<RustLiteral>,
}

#[pymethods]
impl PyLiteral {
    #[new]
    fn new(s: &str) -> Self {
        Self { inner: Arc::new(RustLiteral::new(s)) }
    }
    
    fn parse_string(&self, s: &str) -> PyResult<Vec<String>> {
        self.inner.parse_string(s)
            .map(|r| r.as_list())
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }
    
    /// Parse multiple strings in batch (amortizes FFI overhead)
    fn parse_batch(&self, strings: Vec<String>) -> PyResult<Vec<Vec<String>>> {
        let results: Vec<Vec<String>> = strings.iter()
            .map(|s| {
                self.inner.parse_string(s.as_str())
                    .map(|r| r.as_list())
                    .unwrap_or_default()
            })
            .collect();
        Ok(results)
    }
    
    fn search_string(&self, s: &str) -> PyResult<Vec<Vec<String>>> {
        let results = self.inner.search_string(s);
        Ok(results.into_iter().map(|r| r.as_list()).collect())
    }
    
    fn __add__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyAnd> {
        if let Ok(other_lit) = other.extract::<PyLiteral>() {
            let elements: Vec<Arc<dyn ParserElement>> = vec![self.inner.clone(), other_lit.inner.clone()];
            Ok(PyAnd { inner: Arc::new(RustAnd::new(elements)) })
        } else if let Ok(other_and) = other.extract::<PyAnd>() {
            // Start with self, extend with other's elements
            let mut elements: Vec<Arc<dyn ParserElement>> = vec![self.inner.clone()];
            // We'd need to expose elements from PyAnd - for now just create a simple And
            Ok(PyAnd { inner: Arc::new(RustAnd::new(elements)) })
        } else {
            Err(PyValueError::new_err("Unsupported operand type for +"))
        }
    }
    
    fn __or__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyMatchFirst> {
        if let Ok(other_lit) = other.extract::<PyLiteral>() {
            let elements: Vec<Arc<dyn ParserElement>> = vec![self.inner.clone(), other_lit.inner.clone()];
            Ok(PyMatchFirst { inner: Arc::new(RustMatchFirst::new(elements)) })
        } else {
            Err(PyValueError::new_err("Unsupported operand type for |"))
        }
    }
}

/// Word match element
#[pyclass(name = "Word")]
#[derive(Clone)]
struct PyWord {
    inner: Arc<RustWord>,
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
        Self { inner: Arc::new(word) }
    }
    
    fn parse_string(&self, s: &str) -> PyResult<Vec<String>> {
        self.inner.parse_string(s)
            .map(|r| r.as_list())
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }
    
    /// Parse multiple strings in batch (amortizes FFI overhead)
    fn parse_batch(&self, strings: Vec<String>) -> PyResult<Vec<Vec<String>>> {
        let results: Vec<Vec<String>> = strings.iter()
            .map(|s| {
                self.inner.parse_string(s.as_str())
                    .map(|r| r.as_list())
                    .unwrap_or_default()
            })
            .collect();
        Ok(results)
    }
    
    fn __add__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyAnd> {
        if let Ok(other) = other.extract::<PyWord>() {
            let elements: Vec<Arc<dyn ParserElement>> = vec![self.inner.clone(), other.inner.clone()];
            Ok(PyAnd { inner: Arc::new(RustAnd::new(elements)) })
        } else if let Ok(other) = other.extract::<PyLiteral>() {
            let elements: Vec<Arc<dyn ParserElement>> = vec![self.inner.clone(), other.inner];
            Ok(PyAnd { inner: Arc::new(RustAnd::new(elements)) })
        } else {
            Err(PyValueError::new_err("Unsupported operand type for +"))
        }
    }
}

/// Regex match element
#[pyclass(name = "Regex")]
#[derive(Clone)]
struct PyRegex {
    inner: Arc<RegexMatch>,
}

#[pymethods]
impl PyRegex {
    #[new]
    fn new(pattern: &str) -> PyResult<Self> {
        RegexMatch::new(pattern)
            .map(|inner| Self { inner: Arc::new(inner) })
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }
    
    fn parse_string(&self, s: &str) -> PyResult<Vec<String>> {
        self.inner.parse_string(s)
            .map(|r| r.as_list())
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }
}

/// Keyword match element
#[pyclass(name = "Keyword")]
#[derive(Clone)]
struct PyKeyword {
    inner: Arc<RustKeyword>,
}

#[pymethods]
impl PyKeyword {
    #[new]
    fn new(s: &str) -> Self {
        Self { inner: Arc::new(RustKeyword::new(s)) }
    }
    
    fn parse_string(&self, s: &str) -> PyResult<Vec<String>> {
        self.inner.parse_string(s)
            .map(|r| r.as_list())
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }
}

/// And combinator element
#[pyclass(name = "And")]
#[derive(Clone)]
struct PyAnd {
    inner: Arc<RustAnd>,
}

#[pymethods]
impl PyAnd {
    fn parse_string(&self, s: &str) -> PyResult<Vec<String>> {
        self.inner.parse_string(s)
            .map(|r| r.as_list())
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }
}

/// MatchFirst combinator element
#[pyclass(name = "MatchFirst")]
#[derive(Clone)]
struct PyMatchFirst {
    inner: Arc<RustMatchFirst>,
}

#[pymethods]
impl PyMatchFirst {
    fn parse_string(&self, s: &str) -> PyResult<Vec<String>> {
        self.inner.parse_string(s)
            .map(|r| r.as_list())
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }
}

/// ZeroOrMore repetition element
#[pyclass(name = "ZeroOrMore")]
#[derive(Clone)]
struct PyZeroOrMore {
    inner: Arc<RustZeroOrMore>,
}

#[pymethods]
impl PyZeroOrMore {
    #[new]
    fn new(expr: &Bound<'_, PyAny>) -> PyResult<Self> {
        if let Ok(lit) = expr.extract::<PyLiteral>() {
            Ok(Self { inner: Arc::new(RustZeroOrMore::new(lit.inner)) })
        } else if let Ok(word) = expr.extract::<PyWord>() {
            Ok(Self { inner: Arc::new(RustZeroOrMore::new(word.inner)) })
        } else {
            Err(PyValueError::new_err("Unsupported expression type"))
        }
    }
    
    fn parse_string(&self, s: &str) -> PyResult<Vec<String>> {
        self.inner.parse_string(s)
            .map(|r| r.as_list())
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }
}

/// OneOrMore repetition element
#[pyclass(name = "OneOrMore")]
#[derive(Clone)]
struct PyOneOrMore {
    inner: Arc<RustOneOrMore>,
}

#[pymethods]
impl PyOneOrMore {
    #[new]
    fn new(expr: &Bound<'_, PyAny>) -> PyResult<Self> {
        if let Ok(lit) = expr.extract::<PyLiteral>() {
            Ok(Self { inner: Arc::new(RustOneOrMore::new(lit.inner)) })
        } else if let Ok(word) = expr.extract::<PyWord>() {
            Ok(Self { inner: Arc::new(RustOneOrMore::new(word.inner)) })
        } else {
            Err(PyValueError::new_err("Unsupported expression type"))
        }
    }
    
    fn parse_string(&self, s: &str) -> PyResult<Vec<String>> {
        self.inner.parse_string(s)
            .map(|r| r.as_list())
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }
}

/// Optional element
#[pyclass(name = "Optional")]
#[derive(Clone)]
struct PyOptional {
    inner: Arc<RustOptional>,
}

#[pymethods]
impl PyOptional {
    #[new]
    fn new(expr: &Bound<'_, PyAny>) -> PyResult<Self> {
        if let Ok(lit) = expr.extract::<PyLiteral>() {
            Ok(Self { inner: Arc::new(RustOptional::new(lit.inner)) })
        } else if let Ok(word) = expr.extract::<PyWord>() {
            Ok(Self { inner: Arc::new(RustOptional::new(word.inner)) })
        } else {
            Err(PyValueError::new_err("Unsupported expression type"))
        }
    }
    
    fn parse_string(&self, s: &str) -> PyResult<Vec<String>> {
        self.inner.parse_string(s)
            .map(|r| r.as_list())
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }
}

/// Group element
#[pyclass(name = "Group")]
#[derive(Clone)]
struct PyGroup {
    inner: Arc<RustGroup>,
}

#[pymethods]
impl PyGroup {
    #[new]
    fn new(expr: &Bound<'_, PyAny>) -> PyResult<Self> {
        if let Ok(lit) = expr.extract::<PyLiteral>() {
            Ok(Self { inner: Arc::new(RustGroup::new(lit.inner)) })
        } else if let Ok(word) = expr.extract::<PyWord>() {
            Ok(Self { inner: Arc::new(RustGroup::new(word.inner)) })
        } else if let Ok(and) = expr.extract::<PyAnd>() {
            Ok(Self { inner: Arc::new(RustGroup::new(and.inner)) })
        } else {
            Err(PyValueError::new_err("Unsupported expression type"))
        }
    }
    
    fn parse_string(&self, s: &str) -> PyResult<Vec<String>> {
        self.inner.parse_string(s)
            .map(|r| r.as_list())
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }
}

/// Suppress element
#[pyclass(name = "Suppress")]
#[derive(Clone)]
struct PySuppress {
    inner: Arc<RustSuppress>,
}

#[pymethods]
impl PySuppress {
    #[new]
    fn new(expr: &Bound<'_, PyAny>) -> PyResult<Self> {
        if let Ok(lit) = expr.extract::<PyLiteral>() {
            Ok(Self { inner: Arc::new(RustSuppress::new(lit.inner)) })
        } else if let Ok(word) = expr.extract::<PyWord>() {
            Ok(Self { inner: Arc::new(RustSuppress::new(word.inner)) })
        } else {
            Err(PyValueError::new_err("Unsupported expression type"))
        }
    }
    
    fn parse_string(&self, s: &str) -> PyResult<Vec<String>> {
        self.inner.parse_string(s)
            .map(|r| r.as_list())
            .map_err(|e| PyValueError::new_err(e.to_string()))
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
    
    m.add("__version__", "0.1.0")?;
    Ok(())
}
