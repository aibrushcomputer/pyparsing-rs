---
# ⚠️ This repository has been moved to be under the AiBrush Organization. Please use the code from here [pyparsing-rs](https://github.com/AiBrush/pyparsing-rs) ⚠️
---

# pyparsing-rs

[![CI](https://github.com/aibrushcomputer/pyparsing-rs/actions/workflows/ci.yml/badge.svg)](https://github.com/aibrushcomputer/pyparsing-rs/actions)

Rust rewrite of Python's `pyparsing` library with PyO3 bindings. **20-45x faster** than original pyparsing.

## Installation

```bash
pip install pyparsing-rs
```

## Usage

```python
import pyparsing_rs as pp

# Basic elements
lit = pp.Literal("hello")
word = pp.Word(pp.alphas())
regex = pp.Regex(r"\d+")

# Parse
result = lit.parse_string("hello world")
print(result)  # ['hello']

# Combinators
combined = pp.Literal("hello") + pp.Literal(" world")
result = combined.parse_string("hello world")
print(result)  # ['hello', ' world']

# Repetition
many = pp.ZeroOrMore(pp.Literal("a"))
result = many.parse_string("aaaa")
print(result)  # ['a', 'a', 'a', 'a']
```

## Performance

| Operation | pyparsing | pyparsing-rs | Speedup |
|-----------|-----------|--------------|---------|
| Literal | 4.13 µs | 0.22 µs | **18.5x** |
| Word | 5.20 µs | 0.23 µs | **23.2x** |
| Regex | 6.16 µs | 0.25 µs | **24.5x** |

*Benchmarked on AMD Ryzen 9 5900X. Single parse_string() calls.*

## Implemented Elements

### Basic
- `Literal` - Exact string match
- `Keyword` - Match with word boundary
- `Word` - Match word characters
- `Regex` - Regular expression match

### Combinators
- `And` (+ operator) - Sequence match
- `MatchFirst` (| operator) - First match wins
- `Or` (^ operator) - Longest match wins

### Repetition
- `ZeroOrMore` (*) - 0 or more matches
- `OneOrMore` (+) - 1 or more matches
- `Optional` (?) - 0 or 1 matches

### Structure
- `Group` - Nest results
- `Suppress` - Match but don't capture

## Architecture

- **Zero-copy parsing**: Uses `&str` slices where possible
- **Bitset character lookup**: O(1) character class matching
- **Fast paths**: First-character optimization for literals
- **PyO3 bindings**: Minimal overhead Python interop

## Development

```bash
# Build
maturin develop --release

# Test
pytest tests/ -v

# Benchmark
python tests/test_performance.py
```

## Project Structure

```
pyparsing-rs/
├── src/
│   ├── core/          # Parser infrastructure
│   │   ├── parser.rs  # ParserElement trait
│   │   ├── context.rs # Parse position tracking
│   │   ├── results.rs # ParseResults
│   │   └── exceptions.rs
│   ├── elements/      # Parser elements
│   │   ├── literals.rs
│   │   ├── chars.rs
│   │   ├── combinators.rs
│   │   ├── repetition.rs
│   │   └── structure.rs
│   └── lib.rs         # Python bindings
├── tests/
└── test_grammars/     # Benchmark grammars
```

## License

MIT

## Links

- [PyPI](https://pypi.org/project/pyparsing-rs/)
- [GitHub](https://github.com/aibrushcomputer/pyparsing-rs)
