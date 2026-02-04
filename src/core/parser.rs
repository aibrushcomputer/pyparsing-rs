use crate::core::context::ParseContext;
use crate::core::exceptions::ParseException;
use crate::core::results::ParseResults;

/// Result of a parse attempt
pub type ParseResult<'a> = Result<(usize, ParseResults), ParseException>;

/// Core trait that all parser elements implement
pub trait ParserElement: Send + Sync {
    /// Attempt to parse at the given location
    fn parse_impl<'a>(&self, ctx: &mut ParseContext<'a>, loc: usize) -> ParseResult<'a>;

    /// Get a unique identifier for memoization
    fn parser_id(&self) -> usize;

    /// Human-readable name for error messages
    fn name(&self) -> &str;

    /// Zero-alloc match check — returns end position without creating ParseResults.
    /// Override this for maximum performance on match-only operations.
    fn try_match_at(&self, input: &str, loc: usize) -> Option<usize> {
        let mut ctx = ParseContext::new(input);
        self.parse_impl(&mut ctx, loc).map(|(end, _)| end).ok()
    }

    /// Parse a string from the beginning
    fn parse_string(&self, input: &str) -> Result<ParseResults, ParseException> {
        let mut ctx = ParseContext::new(input);
        let (_, results) = self.parse_impl(&mut ctx, 0)?;
        Ok(results)
    }

    /// Search for matches in a string
    fn search_string(&self, input: &str) -> Vec<ParseResults> {
        let mut ctx = ParseContext::new(input);
        let mut results = Vec::new();
        let mut loc = 0;

        while loc < input.len() {
            match self.parse_impl(&mut ctx, loc) {
                Ok((end_loc, res)) => {
                    results.push(res);
                    loc = end_loc;
                }
                Err(_) => {
                    loc += 1;
                }
            }
        }
        results
    }
}

use std::sync::atomic::{AtomicUsize, Ordering};

static PARSER_ID_COUNTER: AtomicUsize = AtomicUsize::new(0);

pub fn next_parser_id() -> usize {
    PARSER_ID_COUNTER.fetch_add(1, Ordering::Relaxed)
}
