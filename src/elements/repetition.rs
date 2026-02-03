use crate::core::parser::{ParserElement, ParseResult, next_parser_id};
use crate::core::context::ParseContext;
use crate::core::results::ParseResults;
use crate::core::exceptions::ParseException;
use std::sync::Arc;

/// ZeroOrMore - matches 0 or more repetitions
pub struct ZeroOrMore {
    id: usize,
    element: Arc<dyn ParserElement>,
}

impl ZeroOrMore {
    pub fn new(element: Arc<dyn ParserElement>) -> Self {
        Self {
            id: next_parser_id(),
            element,
        }
    }
}

impl ParserElement for ZeroOrMore {
    fn parse_impl<'a>(
        &self,
        ctx: &mut ParseContext<'a>,
        mut loc: usize,
    ) -> ParseResult<'a> {
        let mut results = ParseResults::new();
        
        loop {
            match self.element.parse_impl(ctx, loc) {
                Ok((new_loc, res)) => {
                    if new_loc == loc {
                        // No progress made, prevent infinite loop
                        break;
                    }
                    results.extend(res);
                    loc = new_loc;
                }
                Err(_) => break,
            }
        }
        
        Ok((loc, results))
    }
    
    fn parser_id(&self) -> usize {
        self.id
    }
    
    fn name(&self) -> &str {
        "ZeroOrMore"
    }
}

/// OneOrMore - matches 1 or more repetitions
pub struct OneOrMore {
    id: usize,
    element: Arc<dyn ParserElement>,
}

impl OneOrMore {
    pub fn new(element: Arc<dyn ParserElement>) -> Self {
        Self {
            id: next_parser_id(),
            element,
        }
    }
}

impl ParserElement for OneOrMore {
    fn parse_impl<'a>(
        &self,
        ctx: &mut ParseContext<'a>,
        mut loc: usize,
    ) -> ParseResult<'a> {
        let mut results = ParseResults::new();
        let mut count = 0;
        
        loop {
            match self.element.parse_impl(ctx, loc) {
                Ok((new_loc, res)) => {
                    if new_loc == loc {
                        break;
                    }
                    results.extend(res);
                    loc = new_loc;
                    count += 1;
                }
                Err(_) => break,
            }
        }
        
        if count == 0 {
            Err(ParseException::new(loc, "Expected at least one match"))
        } else {
            Ok((loc, results))
        }
    }
    
    fn parser_id(&self) -> usize {
        self.id
    }
    
    fn name(&self) -> &str {
        "OneOrMore"
    }
}

/// Optional - matches 0 or 1 times
pub struct Optional {
    id: usize,
    element: Arc<dyn ParserElement>,
}

impl Optional {
    pub fn new(element: Arc<dyn ParserElement>) -> Self {
        Self {
            id: next_parser_id(),
            element,
        }
    }
}

impl ParserElement for Optional {
    fn parse_impl<'a>(
        &self,
        ctx: &mut ParseContext<'a>,
        loc: usize,
    ) -> ParseResult<'a> {
        match self.element.parse_impl(ctx, loc) {
            Ok(result) => Ok(result),
            Err(_) => Ok((loc, ParseResults::new())),
        }
    }
    
    fn parser_id(&self) -> usize {
        self.id
    }
    
    fn name(&self) -> &str {
        "Optional"
    }
}

/// Exact repetition - matches exactly n times
pub struct Exactly {
    id: usize,
    element: Arc<dyn ParserElement>,
    count: usize,
}

impl Exactly {
    pub fn new(element: Arc<dyn ParserElement>, count: usize) -> Self {
        Self {
            id: next_parser_id(),
            element,
            count,
        }
    }
}

impl ParserElement for Exactly {
    fn parse_impl<'a>(
        &self,
        ctx: &mut ParseContext<'a>,
        mut loc: usize,
    ) -> ParseResult<'a> {
        let mut results = ParseResults::new();
        
        for _ in 0..self.count {
            match self.element.parse_impl(ctx, loc) {
                Ok((new_loc, res)) => {
                    results.extend(res);
                    loc = new_loc;
                }
                Err(e) => return Err(e),
            }
        }
        
        Ok((loc, results))
    }
    
    fn parser_id(&self) -> usize {
        self.id
    }
    
    fn name(&self) -> &str {
        "Exactly"
    }
}
