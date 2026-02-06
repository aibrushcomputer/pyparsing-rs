use smallvec::SmallVec;
use std::sync::Arc;

/// Parse results that can be accessed as both list and dict
#[derive(Debug, Clone)]
pub struct ParseResults {
    tokens: SmallVec<[Arc<str>; 2]>,
}

impl Default for ParseResults {
    fn default() -> Self {
        Self {
            tokens: SmallVec::new(),
        }
    }
}

impl ParseResults {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn from_single(s: &str) -> Self {
        let mut tokens = SmallVec::new();
        tokens.push(Arc::from(s));
        Self { tokens }
    }

    pub fn extend(&mut self, other: ParseResults) {
        self.tokens.extend(other.tokens);
    }

    pub fn as_vec(&self) -> &[Arc<str>] {
        &self.tokens
    }
}
