use std::fmt;
use std::sync::Arc;

#[derive(Debug, Clone)]
pub struct ParseException {
    pub loc: usize,
    pub msg: Arc<str>,
}

impl ParseException {
    pub fn new(loc: usize, msg: impl Into<Arc<str>>) -> Self {
        Self {
            loc,
            msg: msg.into(),
        }
    }
}

impl fmt::Display for ParseException {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ParseException at position {}: {}", self.loc, self.msg)
    }
}

impl std::error::Error for ParseException {}

#[derive(Debug, Clone)]
pub struct ParseFatalException {
    pub loc: usize,
    pub msg: Arc<str>,
}

impl ParseFatalException {
    pub fn new(loc: usize, msg: impl Into<Arc<str>>) -> Self {
        Self {
            loc,
            msg: msg.into(),
        }
    }
}

impl fmt::Display for ParseFatalException {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "ParseFatalException at position {}: {}",
            self.loc, self.msg
        )
    }
}

impl std::error::Error for ParseFatalException {}
