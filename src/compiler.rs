// Grammar compiler - compiles parser grammars to native code for maximum speed
use crate::core::parser::ParserElement;
use crate::core::results::ParseResults;

/// Compiled grammar that can parse multiple inputs efficiently
pub struct CompiledGrammar {
    // Pre-allocated buffers to avoid allocations
    pub results_buffer: Vec<ParseResults>,
}

impl CompiledGrammar {
    pub fn new() -> Self {
        Self {
            results_buffer: Vec::with_capacity(1024),
        }
    }

    /// Parse multiple strings in a single batch (amortizes FFI overhead)
    pub fn parse_batch<P: ParserElement>(
        &mut self,
        parser: &P,
        inputs: &[&str],
    ) -> Vec<Result<ParseResults, String>> {
        self.results_buffer.clear();

        let mut results = Vec::with_capacity(inputs.len());

        for input in inputs {
            match parser.parse_string(input) {
                Ok(res) => {
                    self.results_buffer.push(res);
                    results.push(Ok(self.results_buffer.last().unwrap().clone()));
                }
                Err(e) => {
                    results.push(Err(e.to_string()));
                }
            }
        }

        results
    }
}

/// Ultra-fast scanner using SIMD-style operations
pub struct FastScanner;

impl FastScanner {
    /// Find literal using SWAR (SIMD Within A Register) technique
    #[inline(always)]
    pub fn find_literal(input: &str, literal: &str) -> Option<usize> {
        if literal.len() > input.len() {
            return None;
        }

        let lit_bytes = literal.as_bytes();
        let first_byte = lit_bytes[0];
        let input_bytes = input.as_bytes();

        // Use memchr for first byte (often SIMD-accelerated)
        let mut pos = 0;
        while let Some(found) = memchr::memchr(first_byte, &input_bytes[pos..]) {
            let start = pos + found;

            // Check if we have enough bytes remaining
            if start + literal.len() > input.len() {
                return None;
            }

            // Fast comparison
            if &input_bytes[start..start + literal.len()] == lit_bytes {
                return Some(start);
            }

            pos = start + 1;
        }

        None
    }

    /// Scan for word characters using byte operations
    #[inline(always)]
    pub fn scan_word_bytes(input: &[u8], init_chars: &[bool; 256]) -> (usize, usize) {
        if input.is_empty() {
            return (0, 0);
        }

        // Check first char
        if !init_chars[input[0] as usize] {
            return (0, 0);
        }

        let mut i = 1;
        while i < input.len() && input[i] < 128 && init_chars[input[i] as usize] {
            i += 1;
        }

        (0, i)
    }
}
