use smallvec::SmallVec;
use std::collections::HashMap;

/// Parse results that can be accessed as both list and dict
#[derive(Debug, Clone)]
pub struct ParseResults {
    tokens: SmallVec<[String; 2]>,
    named: Option<HashMap<String, usize>>,
}

impl Default for ParseResults {
    fn default() -> Self {
        Self {
            tokens: SmallVec::new(),
            named: None,
        }
    }
}

impl ParseResults {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn from_single(s: &str) -> Self {
        let mut tokens = SmallVec::new();
        tokens.push(s.to_string());
        Self {
            tokens,
            named: None,
        }
    }

    pub fn from_vec(tokens: Vec<String>) -> Self {
        Self {
            tokens: SmallVec::from_vec(tokens),
            named: None,
        }
    }

    #[inline(always)]
    pub fn push(&mut self, token: &str) {
        self.tokens.push(token.to_string());
    }

    #[inline(always)]
    pub fn push_string(&mut self, token: String) {
        self.tokens.push(token);
    }

    pub fn set_name(&mut self, name: &str, index: usize) {
        self.named
            .get_or_insert_with(HashMap::new)
            .insert(name.to_string(), index);
    }

    pub fn extend(&mut self, other: ParseResults) {
        let offset = self.tokens.len();
        self.tokens.extend(other.tokens);
        if let Some(other_named) = other.named {
            let named = self.named.get_or_insert_with(HashMap::new);
            for (name, idx) in other_named {
                named.insert(name, idx + offset);
            }
        }
    }

    pub fn as_list(&self) -> Vec<String> {
        self.tokens.to_vec()
    }

    pub fn as_vec(&self) -> &[String] {
        &self.tokens
    }

    pub fn get(&self, index: usize) -> Option<&String> {
        self.tokens.get(index)
    }

    pub fn get_named(&self, name: &str) -> Option<&String> {
        self.named
            .as_ref()
            .and_then(|n| n.get(name))
            .and_then(|&idx| self.tokens.get(idx))
    }

    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }
}
