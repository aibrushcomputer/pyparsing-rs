use crate::core::context::ParseContext;
use crate::core::parser::{next_parser_id, ParseResult, ParserElement};
use crate::core::results::ParseResults;
use std::sync::Arc;

/// Group - wraps results in a nested structure
pub struct Group {
    id: usize,
    element: Arc<dyn ParserElement>,
}

impl Group {
    pub fn new(element: Arc<dyn ParserElement>) -> Self {
        Self {
            id: next_parser_id(),
            element,
        }
    }
}

impl ParserElement for Group {
    fn parse_impl<'a>(&self, ctx: &mut ParseContext<'a>, loc: usize) -> ParseResult<'a> {
        match self.element.parse_impl(ctx, loc) {
            Ok((new_loc, res)) => {
                // Group just passes through the tokens (grouping is semantic at the Python level)
                Ok((new_loc, res))
            }
            Err(e) => Err(e),
        }
    }

    /// Zero-alloc match — delegates to inner element
    #[inline]
    fn try_match_at(&self, input: &str, loc: usize) -> Option<usize> {
        self.element.try_match_at(input, loc)
    }

    fn parser_id(&self) -> usize {
        self.id
    }

    fn name(&self) -> &str {
        "Group"
    }
}

/// Suppress - matches but doesn't add to results
pub struct Suppress {
    id: usize,
    element: Arc<dyn ParserElement>,
}

impl Suppress {
    pub fn new(element: Arc<dyn ParserElement>) -> Self {
        Self {
            id: next_parser_id(),
            element,
        }
    }
}

impl ParserElement for Suppress {
    fn parse_impl<'a>(&self, ctx: &mut ParseContext<'a>, loc: usize) -> ParseResult<'a> {
        // Use try_match_at to avoid creating ParseResults from inner element
        match self.element.try_match_at(ctx.input(), loc) {
            Some(new_loc) => Ok((new_loc, ParseResults::new())),
            None => Err(crate::core::exceptions::ParseException::new(
                loc,
                "Suppress: no match",
            )),
        }
    }

    /// Zero-alloc match — delegates to inner element
    #[inline]
    fn try_match_at(&self, input: &str, loc: usize) -> Option<usize> {
        self.element.try_match_at(input, loc)
    }

    fn parser_id(&self) -> usize {
        self.id
    }

    fn name(&self) -> &str {
        "Suppress"
    }
}
