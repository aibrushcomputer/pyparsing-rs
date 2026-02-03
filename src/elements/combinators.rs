use crate::core::parser::{ParserElement, ParseResult, next_parser_id};
use crate::core::context::ParseContext;
use crate::core::results::ParseResults;
use crate::core::exceptions::ParseException;
use std::sync::Arc;

/// Sequence combinator - all must match in order (And)
pub struct And {
    id: usize,
    elements: Vec<Arc<dyn ParserElement>>,
    name: String,
}

impl And {
    pub fn new(elements: Vec<Arc<dyn ParserElement>>) -> Self {
        let name = format!("And({} elements)", elements.len());
        Self {
            id: next_parser_id(),
            elements,
            name,
        }
    }
    
    pub fn add_element(&mut self, elem: Arc<dyn ParserElement>) {
        self.elements.push(elem);
        self.name = format!("And({} elements)", self.elements.len());
    }
}

impl ParserElement for And {
    fn parse_impl<'a>(
        &self,
        ctx: &mut ParseContext<'a>,
        mut loc: usize,
    ) -> ParseResult<'a> {
        let mut results = ParseResults::new();
        
        for elem in &self.elements {
            match elem.parse_impl(ctx, loc) {
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
        &self.name
    }
}

/// MatchFirst combinator - first match wins (| operator)
pub struct MatchFirst {
    id: usize,
    elements: Vec<Arc<dyn ParserElement>>,
    name: String,
}

impl MatchFirst {
    pub fn new(elements: Vec<Arc<dyn ParserElement>>) -> Self {
        let name = format!("MatchFirst({} elements)", elements.len());
        Self {
            id: next_parser_id(),
            elements,
            name,
        }
    }
    
    pub fn add_element(&mut self, elem: Arc<dyn ParserElement>) {
        self.elements.push(elem);
        self.name = format!("MatchFirst({} elements)", self.elements.len());
    }
}

impl ParserElement for MatchFirst {
    fn parse_impl<'a>(
        &self,
        ctx: &mut ParseContext<'a>,
        loc: usize,
    ) -> ParseResult<'a> {
        let mut last_error = None;
        
        for elem in &self.elements {
            match elem.parse_impl(ctx, loc) {
                Ok(result) => return Ok(result),
                Err(e) => last_error = Some(e),
            }
        }
        
        Err(last_error.unwrap_or_else(|| ParseException::new(loc, "No match found")))
    }
    
    fn parser_id(&self) -> usize {
        self.id
    }
    
    fn name(&self) -> &str {
        &self.name
    }
}

/// Or combinator - longest match wins (^ operator)
pub struct Or {
    id: usize,
    elements: Vec<Arc<dyn ParserElement>>,
    name: String,
}

impl Or {
    pub fn new(elements: Vec<Arc<dyn ParserElement>>) -> Self {
        let name = format!("Or({} elements)", elements.len());
        Self {
            id: next_parser_id(),
            elements,
            name,
        }
    }
}

impl ParserElement for Or {
    fn parse_impl<'a>(
        &self,
        ctx: &mut ParseContext<'a>,
        loc: usize,
    ) -> ParseResult<'a> {
        let mut best_result: Option<(usize, ParseResults)> = None;
        
        for elem in &self.elements {
            match elem.parse_impl(ctx, loc) {
                Ok((new_loc, res)) => {
                    // Keep the longest match
                    if best_result.is_none() || new_loc > best_result.as_ref().unwrap().0 {
                        best_result = Some((new_loc, res));
                    }
                }
                Err(_) => {}
            }
        }
        
        match best_result {
            Some(result) => Ok(result),
            None => Err(ParseException::new(loc, "No match found")),
        }
    }
    
    fn parser_id(&self) -> usize {
        self.id
    }
    
    fn name(&self) -> &str {
        &self.name
    }
}
