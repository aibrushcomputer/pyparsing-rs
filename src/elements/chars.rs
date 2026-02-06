use crate::core::context::ParseContext;
use crate::core::exceptions::ParseException;
use crate::core::parser::{next_parser_id, ParseResult, ParserElement};
use crate::core::results::ParseResults;
use std::sync::Arc;

/// 256-bit bitset for O(1) character lookup
#[derive(Clone)]
pub struct CharSet {
    bits: [u64; 4], // 256 bits total
}

impl CharSet {
    pub fn from_chars(chars: &str) -> Self {
        let mut bits = [0u64; 4];
        for c in chars.chars() {
            let c = c as usize;
            if c < 256 {
                bits[c / 64] |= 1u64 << (c % 64);
            }
        }
        Self { bits }
    }

    #[inline(always)]
    pub fn contains(&self, c: u8) -> bool {
        let c = c as usize;
        (self.bits[c / 64] >> (c % 64)) & 1 != 0
    }

    #[inline(always)]
    pub fn contains_char(&self, c: char) -> bool {
        let c = c as u32;
        if c >= 256 {
            return false;
        }
        self.contains(c as u8)
    }
}

/// Match a word made up of characters from specified set
pub struct Word {
    id: usize,
    init_chars: CharSet,
    body_chars: CharSet,
    min_len: usize,
    max_len: usize,
    name: String,
    error_msg: Arc<str>,
}

impl Word {
    pub fn new(init_chars: &str) -> Self {
        let charset = CharSet::from_chars(init_chars);
        let name = format!("W:({}...)", &init_chars[..init_chars.len().min(8)]);
        let error_msg: Arc<str> = format!("Expected {}", name).into();

        Self {
            id: next_parser_id(),
            init_chars: charset.clone(),
            body_chars: charset,
            min_len: 1,
            max_len: 0, // 0 means unlimited
            name,
            error_msg,
        }
    }

    pub fn with_body_chars(mut self, body: &str) -> Self {
        self.body_chars = CharSet::from_chars(body);
        self
    }

    #[inline(always)]
    pub fn init_chars_contains(&self, b: u8) -> bool {
        self.init_chars.contains(b)
    }

    #[inline(always)]
    pub fn body_chars_contains(&self, b: u8) -> bool {
        self.body_chars.contains(b)
    }
}

impl ParserElement for Word {
    #[inline]
    fn parse_impl<'a>(&self, _ctx: &mut ParseContext<'a>, loc: usize) -> ParseResult<'a> {
        let input = _ctx.input();

        if loc >= input.len() {
            return Err(ParseException::new(loc, self.error_msg.clone()));
        }

        // Check first character (ASCII fast path)
        let first_byte = input.as_bytes()[loc];
        if !self.init_chars.contains(first_byte) {
            return Err(ParseException::new(loc, self.error_msg.clone()));
        }

        // Find end of word using byte scan
        let mut end = loc + 1;
        let bytes = input.as_bytes();

        while end < bytes.len() {
            let b = bytes[end];
            // Fast ASCII check
            if b < 128 {
                if !self.body_chars.contains(b) {
                    break;
                }
                end += 1;
            } else {
                // UTF-8 handling
                let c = input[end..].chars().next().unwrap();
                if !self.body_chars.contains_char(c) {
                    break;
                }
                end += c.len_utf8();
            }

            if self.max_len > 0 && end - loc >= self.max_len {
                break;
            }
        }

        // Check minimum length (byte count == char count for ASCII)
        if self.min_len > 0 {
            let byte_len = end - loc;
            if byte_len < self.min_len {
                return Err(ParseException::new(loc, self.error_msg.clone()));
            }
            // Only do expensive char count for non-ASCII content
            if byte_len >= self.min_len {
                // For ASCII-only content, byte_len == char_count, already checked
                // For multi-byte chars, we need the actual count
                let all_ascii = bytes[loc..end].iter().all(|&b| b < 128);
                if !all_ascii {
                    let char_count = input[loc..end].chars().count();
                    if char_count < self.min_len {
                        return Err(ParseException::new(loc, self.error_msg.clone()));
                    }
                }
            }
        }

        let matched = &input[loc..end];
        Ok((end, ParseResults::from_single(matched)))
    }

    /// Zero-alloc match — just returns end position, no ParseResults
    #[inline]
    fn try_match_at(&self, input: &str, loc: usize) -> Option<usize> {
        let bytes = input.as_bytes();
        if loc >= bytes.len() || !self.init_chars.contains(bytes[loc]) {
            return None;
        }
        let mut end = loc + 1;
        while end < bytes.len() && self.body_chars.contains(bytes[end]) {
            end += 1;
        }
        if self.max_len > 0 && end - loc > self.max_len {
            end = loc + self.max_len;
        }
        // Check min_len — must match at least this many characters
        if end - loc < self.min_len {
            return None;
        }
        Some(end)
    }

    fn parser_id(&self) -> usize {
        self.id
    }

    /// Optimized search: scan bytes directly without ParseContext overhead
    fn search_string(&self, input: &str) -> Vec<ParseResults> {
        let bytes = input.as_bytes();
        let len = bytes.len();
        let mut results = Vec::new();
        let mut pos = 0;

        while pos < len {
            let b = bytes[pos];
            if b < 128 && self.init_chars.contains(b) {
                let start = pos;
                pos += 1;
                while pos < len {
                    let b2 = bytes[pos];
                    if b2 < 128 {
                        if !self.body_chars.contains(b2) {
                            break;
                        }
                        pos += 1;
                    } else {
                        let c = input[pos..].chars().next().unwrap();
                        if !self.body_chars.contains_char(c) {
                            break;
                        }
                        pos += c.len_utf8();
                    }
                }
                results.push(ParseResults::from_single(&input[start..pos]));
            } else if b >= 128 {
                let c = input[pos..].chars().next().unwrap();
                if self.init_chars.contains_char(c) {
                    let start = pos;
                    pos += c.len_utf8();
                    while pos < len {
                        let b2 = bytes[pos];
                        if b2 < 128 {
                            if !self.body_chars.contains(b2) {
                                break;
                            }
                            pos += 1;
                        } else {
                            let c2 = input[pos..].chars().next().unwrap();
                            if !self.body_chars.contains_char(c2) {
                                break;
                            }
                            pos += c2.len_utf8();
                        }
                    }
                    results.push(ParseResults::from_single(&input[start..pos]));
                } else {
                    pos += c.len_utf8();
                }
            } else {
                pos += 1;
            }
        }
        results
    }

    fn name(&self) -> &str {
        &self.name
    }
}

/// Fast-path category for common regex patterns
enum FastPath {
    /// \s+ — one or more whitespace
    WhitespacePlus,
    /// Single-char class like [+\-*/] — stored as 256-bit lookup
    SingleCharClass(CharSet),
    /// No fast path, use regex engine
    None,
}

fn detect_fast_path(pattern: &str) -> FastPath {
    // Check for \s+ pattern
    if pattern == r"\s+" || pattern == r"\s*" {
        return FastPath::WhitespacePlus;
    }
    // Check for [chars] single-char class (no quantifiers, no nested brackets)
    if pattern.starts_with('[')
        && pattern.ends_with(']')
        && !pattern[1..pattern.len() - 1].contains('[')
    {
        let inner = &pattern[1..pattern.len() - 1];
        let mut chars = String::new();
        let mut escape = false;
        for c in inner.chars() {
            if escape {
                chars.push(c);
                escape = false;
            } else if c == '\\' {
                escape = true;
            } else {
                chars.push(c);
            }
        }
        if !chars.is_empty() {
            return FastPath::SingleCharClass(CharSet::from_chars(&chars));
        }
    }
    FastPath::None
}

/// Match using a regular expression
pub struct RegexMatch {
    id: usize,
    pattern: regex::Regex,
    /// Unanchored version for search_string / find_iter operations
    search_pattern: regex::Regex,
    pattern_str: String,
    error_msg: Arc<str>,
    fast_path: FastPath,
}

impl RegexMatch {
    pub fn new(pattern: &str) -> Result<Self, regex::Error> {
        let anchored = if pattern.starts_with('^') {
            pattern.to_string()
        } else {
            format!("^(?:{})", pattern)
        };
        // Unanchored pattern for search operations (find_iter)
        let unanchored = format!("(?:{})", pattern);

        let error_msg: Arc<str> = format!("Expected match for /{}/", pattern).into();
        let fast_path = detect_fast_path(pattern);

        Ok(Self {
            id: next_parser_id(),
            pattern: regex::Regex::new(&anchored)?,
            search_pattern: regex::Regex::new(&unanchored)?,
            pattern_str: pattern.to_string(),
            error_msg,
            fast_path,
        })
    }

    /// Direct regex match without ParseContext overhead — returns matched substring
    #[inline]
    pub fn try_match<'a>(&self, input: &'a str) -> Option<&'a str> {
        self.pattern.find(input).map(|m| m.as_str())
    }

    /// Iterator over all non-overlapping matches in a haystack
    #[inline]
    pub fn find_iter<'r, 'h>(&'r self, haystack: &'h str) -> regex::Matches<'r, 'h> {
        self.search_pattern.find_iter(haystack)
    }
}

impl ParserElement for RegexMatch {
    #[inline]
    fn parse_impl<'a>(&self, _ctx: &mut ParseContext<'a>, loc: usize) -> ParseResult<'a> {
        let input = &_ctx.input()[loc..];

        if let Some(m) = self.pattern.find(input) {
            let matched = m.as_str();
            Ok((loc + matched.len(), ParseResults::from_single(matched)))
        } else {
            Err(ParseException::new(loc, self.error_msg.clone()))
        }
    }

    /// Zero-alloc match — fast path for common patterns, regex fallback
    #[inline]
    fn try_match_at(&self, input: &str, loc: usize) -> Option<usize> {
        let bytes = input.as_bytes();
        match &self.fast_path {
            FastPath::WhitespacePlus => {
                if loc >= bytes.len() || !bytes[loc].is_ascii_whitespace() {
                    return None;
                }
                let mut end = loc + 1;
                while end < bytes.len() && bytes[end].is_ascii_whitespace() {
                    end += 1;
                }
                Some(end)
            }
            FastPath::SingleCharClass(cs) => {
                if loc >= bytes.len() || !cs.contains(bytes[loc]) {
                    return None;
                }
                Some(loc + 1)
            }
            FastPath::None => self.pattern.find(&input[loc..]).map(|m| loc + m.end()),
        }
    }

    /// Optimized search using the unanchored search_pattern
    fn search_string(&self, input: &str) -> Vec<ParseResults> {
        self.search_pattern
            .find_iter(input)
            .map(|m| ParseResults::from_single(m.as_str()))
            .collect()
    }

    fn parser_id(&self) -> usize {
        self.id
    }

    fn name(&self) -> &str {
        &self.pattern_str
    }
}
