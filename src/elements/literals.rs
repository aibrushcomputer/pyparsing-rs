use crate::core::parser::{ParserElement, ParseResult, next_parser_id};
use crate::core::context::ParseContext;
use crate::core::results::ParseResults;
use crate::core::exceptions::ParseException;

/// Match an exact literal string
pub struct Literal {
    id: usize,
    match_string: String,
}

impl Literal {
    pub fn new(s: &str) -> Self {
        Self {
            id: next_parser_id(),
            match_string: s.to_string(),
        }
    }
}

impl ParserElement for Literal {
    #[inline(always)]
    fn parse_impl<'a>(
        &self,
        _ctx: &mut ParseContext<'a>,
        loc: usize,
    ) -> ParseResult<'a> {
        let input = _ctx.input();
        let match_len = self.match_string.len();
        
        // Fast path: check length first
        if loc + match_len > input.len() {
            return Err(ParseException::new(
                loc,
                format!("Expected '{}'", self.match_string),
            ));
        }
        
        // Fast byte comparison for ASCII literals
        let input_bytes = input.as_bytes();
        let match_bytes = self.match_string.as_bytes();
        
        if input_bytes[loc..loc + match_len] == match_bytes[..] {
            let results = ParseResults::from_single(&self.match_string);
            Ok((loc + match_len, results))
        } else {
            Err(ParseException::new(
                loc,
                format!("Expected '{}'", self.match_string),
            ))
        }
    }
    
    fn parser_id(&self) -> usize {
        self.id
    }
    
    fn name(&self) -> &str {
        &self.match_string
    }
}

/// Match a keyword (literal with word boundary checking)
pub struct Keyword {
    id: usize,
    match_string: String,
    ident_chars: Vec<char>,
}

impl Keyword {
    pub fn new(s: &str) -> Self {
        let ident_chars: Vec<char> = 
            "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_"
            .chars().collect();
        
        Self {
            id: next_parser_id(),
            match_string: s.to_string(),
            ident_chars,
        }
    }
}

impl ParserElement for Keyword {
    #[inline]
    fn parse_impl<'a>(
        &self,
        _ctx: &mut ParseContext<'a>,
        loc: usize,
    ) -> ParseResult<'a> {
        let input = _ctx.input();
        let match_len = self.match_string.len();
        let end_loc = loc + match_len;
        
        if end_loc <= input.len() 
            && &input[loc..end_loc] == self.match_string 
        {
            // Check word boundary after
            if end_loc < input.len() {
                let next_char = input[end_loc..].chars().next().unwrap();
                if self.ident_chars.contains(&next_char) {
                    return Err(ParseException::new(
                        loc,
                        format!("Expected keyword '{}'", self.match_string),
                    ));
                }
            }
            
            let results = ParseResults::from_single(&self.match_string);
            Ok((end_loc, results))
        } else {
            Err(ParseException::new(
                loc,
                format!("Expected keyword '{}'", self.match_string),
            ))
        }
    }
    
    fn parser_id(&self) -> usize {
        self.id
    }
    
    fn name(&self) -> &str {
        &self.match_string
    }
}
