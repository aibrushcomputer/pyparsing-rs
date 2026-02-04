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

        // Check minimum length
        if self.min_len > 0 {
            let char_count = input[loc..end].chars().count();
            if char_count < self.min_len {
                return Err(ParseException::new(loc, self.error_msg.clone()));
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
        Some(end)
    }

    fn parser_id(&self) -> usize {
        self.id
    }

    fn name(&self) -> &str {
        &self.name
    }
}

/// Match using a regular expression
pub struct RegexMatch {
    id: usize,
    pattern: regex::Regex,
    pattern_str: String,
    error_msg: Arc<str>,
}

impl RegexMatch {
    pub fn new(pattern: &str) -> Result<Self, regex::Error> {
        let anchored = if pattern.starts_with('^') {
            pattern.to_string()
        } else {
            format!("^(?:{})", pattern)
        };

        let error_msg: Arc<str> = format!("Expected match for /{}/", pattern).into();

        Ok(Self {
            id: next_parser_id(),
            pattern: regex::Regex::new(&anchored)?,
            pattern_str: pattern.to_string(),
            error_msg,
        })
    }

    /// Direct regex match without ParseContext overhead — returns matched substring
    #[inline]
    pub fn try_match<'a>(&self, input: &'a str) -> Option<&'a str> {
        self.pattern.find(input).map(|m| m.as_str())
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

    /// Zero-alloc match — just returns end position
    #[inline]
    fn try_match_at(&self, input: &str, loc: usize) -> Option<usize> {
        self.pattern.find(&input[loc..]).map(|m| loc + m.end())
    }

    fn parser_id(&self) -> usize {
        self.id
    }

    fn name(&self) -> &str {
        &self.pattern_str
    }
}
