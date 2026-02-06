use crate::core::context::ParseContext;
use crate::core::exceptions::ParseException;
use crate::core::results::ParseResults;

/// Result of a parse attempt
pub type ParseResult<'a> = Result<(usize, ParseResults), ParseException>;

/// Core trait that all parser elements implement
pub trait ParserElement: Send + Sync {
    /// Attempt to parse at the given location
    fn parse_impl<'a>(&self, ctx: &mut ParseContext<'a>, loc: usize) -> ParseResult<'a>;

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

    /// Search for matches in a string.
    /// Default uses try_match_at as pre-filter to avoid parse_impl allocations at
    /// non-matching positions.
    fn search_string(&self, input: &str) -> Vec<ParseResults> {
        let mut ctx = ParseContext::new(input);
        let mut results = Vec::new();
        let mut loc = 0;

        while loc < input.len() {
            // Fast pre-check: does any match start here?
            if let Some(end) = self.try_match_at(input, loc) {
                // Full parse to get token breakdown
                match self.parse_impl(&mut ctx, loc) {
                    Ok((end_loc, res)) => {
                        results.push(res);
                        loc = end_loc;
                    }
                    Err(_) => {
                        // try_match_at agreed but parse_impl disagreed (rare)
                        loc = end.max(loc + 1);
                    }
                }
            } else {
                loc += 1;
            }
        }
        results
    }
}
