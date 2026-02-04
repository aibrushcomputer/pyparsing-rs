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
                // Create a nested group
                let mut grouped = ParseResults::new();
                // Add all tokens as a single group
                for token in res.as_list() {
                    grouped.push(&token);
                }
                Ok((new_loc, grouped))
            }
            Err(e) => Err(e),
        }
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
        match self.element.parse_impl(ctx, loc) {
            Ok((new_loc, _)) => {
                // Return empty results
                Ok((new_loc, ParseResults::new()))
            }
            Err(e) => Err(e),
        }
    }

    fn parser_id(&self) -> usize {
        self.id
    }

    fn name(&self) -> &str {
        "Suppress"
    }
}
