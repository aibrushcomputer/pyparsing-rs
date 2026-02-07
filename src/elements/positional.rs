use crate::core::context::ParseContext;
use crate::core::exceptions::ParseException;
use crate::core::parser::{ParseResult, ParserElement};
use crate::core::results::ParseResults;
use std::sync::Arc;

/// Matches at the start of the string (position 0 only).
pub struct StringStart;

impl ParserElement for StringStart {
    fn parse_impl<'a>(&self, _ctx: &mut ParseContext<'a>, loc: usize) -> ParseResult<'a> {
        if loc == 0 {
            Ok((0, ParseResults::new()))
        } else {
            Err(ParseException::new(loc, "Expected start of string"))
        }
    }

    #[inline(always)]
    fn try_match_at(&self, _input: &str, loc: usize) -> Option<usize> {
        if loc == 0 {
            Some(0)
        } else {
            None
        }
    }
}

/// Matches at the end of the string.
pub struct StringEnd;

impl ParserElement for StringEnd {
    fn parse_impl<'a>(&self, ctx: &mut ParseContext<'a>, loc: usize) -> ParseResult<'a> {
        if loc >= ctx.input().len() {
            Ok((loc, ParseResults::new()))
        } else {
            Err(ParseException::new(loc, "Expected end of string"))
        }
    }

    #[inline(always)]
    fn try_match_at(&self, input: &str, loc: usize) -> Option<usize> {
        if loc >= input.len() {
            Some(loc)
        } else {
            None
        }
    }
}

/// Matches at the start of a line (position 0 or after \n).
pub struct LineStart;

impl ParserElement for LineStart {
    fn parse_impl<'a>(&self, ctx: &mut ParseContext<'a>, loc: usize) -> ParseResult<'a> {
        if loc == 0 || ctx.input().as_bytes().get(loc.wrapping_sub(1)) == Some(&b'\n') {
            Ok((loc, ParseResults::new()))
        } else {
            Err(ParseException::new(loc, "Expected start of line"))
        }
    }

    #[inline(always)]
    fn try_match_at(&self, input: &str, loc: usize) -> Option<usize> {
        if loc == 0 || input.as_bytes().get(loc.wrapping_sub(1)) == Some(&b'\n') {
            Some(loc)
        } else {
            None
        }
    }
}

/// Matches at the end of a line (before \n or at end of string).
pub struct LineEnd;

impl ParserElement for LineEnd {
    fn parse_impl<'a>(&self, ctx: &mut ParseContext<'a>, loc: usize) -> ParseResult<'a> {
        let input = ctx.input();
        if loc >= input.len() || input.as_bytes()[loc] == b'\n' {
            // Consume the newline if present
            let new_loc = if loc < input.len() { loc + 1 } else { loc };
            Ok((new_loc, ParseResults::from_single("\n")))
        } else {
            Err(ParseException::new(loc, "Expected end of line"))
        }
    }

    #[inline(always)]
    fn try_match_at(&self, input: &str, loc: usize) -> Option<usize> {
        if loc >= input.len() || input.as_bytes()[loc] == b'\n' {
            Some(if loc < input.len() { loc + 1 } else { loc })
        } else {
            None
        }
    }
}

/// Matches the rest of the line (up to but not including the next newline).
pub struct RestOfLine {
    error_msg: Arc<str>,
}

impl RestOfLine {
    pub fn new() -> Self {
        Self {
            error_msg: Arc::from("Expected rest of line"),
        }
    }
}

impl ParserElement for RestOfLine {
    fn parse_impl<'a>(&self, ctx: &mut ParseContext<'a>, loc: usize) -> ParseResult<'a> {
        let input = ctx.input();
        if loc > input.len() {
            return Err(ParseException::new(loc, self.error_msg.clone()));
        }
        let rest = &input[loc..];
        let end = rest.find('\n').map(|p| loc + p).unwrap_or(input.len());
        Ok((end, ParseResults::from_single(&input[loc..end])))
    }

    #[inline(always)]
    fn try_match_at(&self, input: &str, loc: usize) -> Option<usize> {
        if loc > input.len() {
            return None;
        }
        let rest = &input[loc..];
        Some(rest.find('\n').map(|p| loc + p).unwrap_or(input.len()))
    }
}
