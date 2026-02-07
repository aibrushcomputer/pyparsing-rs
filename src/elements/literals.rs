use crate::core::context::ParseContext;
use crate::core::exceptions::ParseException;
use crate::core::parser::{ParseResult, ParserElement};
use crate::core::results::ParseResults;
use std::sync::Arc;

/// Match a single character from a set of characters
pub struct Char {
    charset: [bool; 256],
    error_msg: Arc<str>,
}

impl Char {
    pub fn new(chars: &str) -> Self {
        let mut charset = [false; 256];
        for b in chars.bytes() {
            charset[b as usize] = true;
        }
        Self {
            charset,
            error_msg: Arc::from(format!("Expected one of '{}'", chars)),
        }
    }
}

impl ParserElement for Char {
    fn parse_impl<'a>(&self, ctx: &mut ParseContext<'a>, loc: usize) -> ParseResult<'a> {
        let input = ctx.input();
        if loc < input.len() && self.charset[input.as_bytes()[loc] as usize] {
            Ok((loc + 1, ParseResults::from_single(&input[loc..loc + 1])))
        } else {
            Err(ParseException::new(loc, self.error_msg.clone()))
        }
    }

    #[inline(always)]
    fn try_match_at(&self, input: &str, loc: usize) -> Option<usize> {
        if loc < input.len() && self.charset[input.as_bytes()[loc] as usize] {
            Some(loc + 1)
        } else {
            None
        }
    }
}

/// Match an exact literal string
pub struct Literal {
    match_string: String,
    first_char: u8,
    error_msg: Arc<str>,
    cached_result: ParseResults,
}

impl Literal {
    pub fn new(s: &str) -> Self {
        let first_char = s.bytes().next().unwrap_or(0);
        let error_msg: Arc<str> = format!("Expected '{}'", s).into();
        let cached_result = ParseResults::from_single(s);
        Self {
            match_string: s.to_string(),
            first_char,
            error_msg,
            cached_result,
        }
    }

    #[inline(always)]
    pub fn match_str(&self) -> &str {
        &self.match_string
    }

    #[inline(always)]
    pub fn first_byte(&self) -> u8 {
        self.first_char
    }
}

impl ParserElement for Literal {
    #[inline(always)]
    fn parse_impl<'a>(&self, _ctx: &mut ParseContext<'a>, loc: usize) -> ParseResult<'a> {
        let input = _ctx.input();
        let match_len = self.match_string.len();

        // Fast path: check length first
        if loc + match_len > input.len() {
            return Err(ParseException::new(loc, self.error_msg.clone()));
        }

        // Fast byte comparison
        let input_bytes = input.as_bytes();

        // Check first byte quickly
        if input_bytes[loc] != self.first_char {
            return Err(ParseException::new(loc, self.error_msg.clone()));
        }

        // Check remaining bytes
        let match_bytes = self.match_string.as_bytes();
        if match_len > 1 && input_bytes[loc + 1..loc + match_len] != match_bytes[1..] {
            return Err(ParseException::new(loc, self.error_msg.clone()));
        }

        Ok((loc + match_len, self.cached_result.clone()))
    }

    /// Zero-alloc match — just returns end position
    #[inline(always)]
    fn try_match_at(&self, input: &str, loc: usize) -> Option<usize> {
        let match_len = self.match_string.len();
        let bytes = input.as_bytes();
        let match_bytes = self.match_string.as_bytes();
        if loc + match_len <= bytes.len()
            && bytes[loc] == self.first_char
            && bytes[loc..loc + match_len] == *match_bytes
        {
            Some(loc + match_len)
        } else {
            None
        }
    }
}

/// Match a keyword (literal with word boundary checking)
pub struct Keyword {
    match_string: String,
    match_len: usize,
    first_char: u8,
    ident_chars: [bool; 256],
    error_msg: Arc<str>,
    cached_result: ParseResults,
}

impl Keyword {
    pub fn new(s: &str) -> Self {
        let mut ident_chars = [false; 256];
        for c in b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_" {
            ident_chars[*c as usize] = true;
        }

        let first_char = s.bytes().next().unwrap_or(0);
        let error_msg: Arc<str> = format!("Expected keyword '{}'", s).into();
        let cached_result = ParseResults::from_single(s);

        Self {
            match_string: s.to_string(),
            match_len: s.len(),
            first_char,
            ident_chars,
            error_msg,
            cached_result,
        }
    }
}

impl ParserElement for Keyword {
    #[inline]
    fn parse_impl<'a>(&self, _ctx: &mut ParseContext<'a>, loc: usize) -> ParseResult<'a> {
        let input = _ctx.input();
        let end_loc = loc + self.match_len;

        // Fast checks first
        if end_loc > input.len() {
            return Err(ParseException::new(loc, self.error_msg.clone()));
        }

        let input_bytes = input.as_bytes();

        // Quick first char check
        if input_bytes[loc] != self.first_char {
            return Err(ParseException::new(loc, self.error_msg.clone()));
        }

        // Check rest of string
        if self.match_len > 1 && input[loc + 1..end_loc] != self.match_string[1..] {
            return Err(ParseException::new(loc, self.error_msg.clone()));
        }

        // Check word boundary after
        if end_loc < input.len() {
            let next_byte = input_bytes[end_loc];
            if self.ident_chars[next_byte as usize] {
                return Err(ParseException::new(loc, self.error_msg.clone()));
            }
        }

        Ok((end_loc, self.cached_result.clone()))
    }

    /// Zero-alloc keyword match with word boundary check
    #[inline(always)]
    fn try_match_at(&self, input: &str, loc: usize) -> Option<usize> {
        let end_loc = loc + self.match_len;
        let bytes = input.as_bytes();
        let match_bytes = self.match_string.as_bytes();

        if end_loc > bytes.len()
            || bytes[loc] != self.first_char
            || (self.match_len > 1 && bytes[loc + 1..end_loc] != match_bytes[1..])
        {
            return None;
        }

        // Word boundary check
        if end_loc < bytes.len() && self.ident_chars[bytes[end_loc] as usize] {
            return None;
        }

        Some(end_loc)
    }
}

/// Case-insensitive literal match. Returns the match string in its original case
/// (as specified at construction), not the case found in the input.
pub struct CaselessLiteral {
    match_lower: String,
    error_msg: Arc<str>,
    cached_result: ParseResults,
}

impl CaselessLiteral {
    pub fn new(s: &str) -> Self {
        let match_lower = s.to_ascii_lowercase();
        let error_msg: Arc<str> = format!("Expected '{}' (caseless)", s).into();
        let cached_result = ParseResults::from_single(s);
        Self {
            match_lower,
            error_msg,
            cached_result,
        }
    }
}

impl ParserElement for CaselessLiteral {
    fn parse_impl<'a>(&self, ctx: &mut ParseContext<'a>, loc: usize) -> ParseResult<'a> {
        let input = ctx.input();
        let match_len = self.match_lower.len();
        if loc + match_len > input.len() {
            return Err(ParseException::new(loc, self.error_msg.clone()));
        }
        if input.as_bytes()[loc..loc + match_len]
            .iter()
            .zip(self.match_lower.as_bytes())
            .all(|(a, b)| a.to_ascii_lowercase() == *b)
        {
            Ok((loc + match_len, self.cached_result.clone()))
        } else {
            Err(ParseException::new(loc, self.error_msg.clone()))
        }
    }

    #[inline(always)]
    fn try_match_at(&self, input: &str, loc: usize) -> Option<usize> {
        let match_len = self.match_lower.len();
        if loc + match_len > input.len() {
            return None;
        }
        if input.as_bytes()[loc..loc + match_len]
            .iter()
            .zip(self.match_lower.as_bytes())
            .all(|(a, b)| a.to_ascii_lowercase() == *b)
        {
            Some(loc + match_len)
        } else {
            None
        }
    }
}

/// Case-insensitive keyword match with word boundary checking.
pub struct CaselessKeyword {
    match_lower: String,
    match_len: usize,
    ident_chars: [bool; 256],
    error_msg: Arc<str>,
    cached_result: ParseResults,
}

impl CaselessKeyword {
    pub fn new(s: &str) -> Self {
        let mut ident_chars = [false; 256];
        for c in b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_" {
            ident_chars[*c as usize] = true;
        }
        let match_lower = s.to_ascii_lowercase();
        let error_msg: Arc<str> = format!("Expected keyword '{}' (caseless)", s).into();
        let cached_result = ParseResults::from_single(s);
        Self {
            match_lower,
            match_len: s.len(),
            ident_chars,
            error_msg,
            cached_result,
        }
    }
}

impl ParserElement for CaselessKeyword {
    fn parse_impl<'a>(&self, ctx: &mut ParseContext<'a>, loc: usize) -> ParseResult<'a> {
        let input = ctx.input();
        let end_loc = loc + self.match_len;
        if end_loc > input.len() {
            return Err(ParseException::new(loc, self.error_msg.clone()));
        }
        // Case-insensitive compare
        if !input.as_bytes()[loc..end_loc]
            .iter()
            .zip(self.match_lower.as_bytes())
            .all(|(a, b)| a.to_ascii_lowercase() == *b)
        {
            return Err(ParseException::new(loc, self.error_msg.clone()));
        }
        // Word boundary check
        if end_loc < input.len() && self.ident_chars[input.as_bytes()[end_loc] as usize] {
            return Err(ParseException::new(loc, self.error_msg.clone()));
        }
        Ok((end_loc, self.cached_result.clone()))
    }

    #[inline(always)]
    fn try_match_at(&self, input: &str, loc: usize) -> Option<usize> {
        let end_loc = loc + self.match_len;
        let bytes = input.as_bytes();
        if end_loc > bytes.len() {
            return None;
        }
        if !bytes[loc..end_loc]
            .iter()
            .zip(self.match_lower.as_bytes())
            .all(|(a, b)| a.to_ascii_lowercase() == *b)
        {
            return None;
        }
        if end_loc < bytes.len() && self.ident_chars[bytes[end_loc] as usize] {
            return None;
        }
        Some(end_loc)
    }
}
