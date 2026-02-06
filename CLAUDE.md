# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

pyparsing-rs is a Rust rewrite of Python's `pyparsing` parser combinator library using PyO3 bindings. It targets API compatibility with pyparsing's core parsing operations while achieving 20-45x speedup.

## Build & Test Commands

```bash
# Build (installs as importable Python module in current env)
maturin develop --release

# Run all tests
pytest tests/ -v

# Run a single test file
pytest tests/test_basic.py -v

# Run a single test
pytest tests/test_basic.py::test_literal -v

# Run performance benchmarks
python tests/test_performance.py

# Build wheel (for distribution)
maturin build --release
```

Requires: Python >= 3.9, Rust stable toolchain, `maturin` and `pytest` pip packages.

## Architecture

**Rust core** (`src/`) with **Python bindings** via PyO3. The library compiles to a `cdylib` that Python imports as `pyparsing_rs`.

### Core layer (`src/core/`)
- `parser.rs` — `ParserElement` trait: the base interface all parsers implement. Key methods: `parse_impl()` (internal parse at location), `parse_string()` (public entry point), `search_string()` (find all matches). Each parser gets a unique atomic ID via `next_parser_id()` for memoization.
- `context.rs` — `ParseContext`: holds input string reference and parse position (zero-copy).
- `results.rs` — `ParseResults`: token collection with optional named captures.
- `exceptions.rs` — `ParseException` and `ParseFatalException`.

### Parser elements (`src/elements/`)
Each file implements `ParserElement` for a category of parsers:
- `literals.rs` — `Literal`, `Keyword` (exact match, keyword with word boundary)
- `chars.rs` — `Word` (character class matching via 256-bit `CharSet` bitset), `Regex`
- `combinators.rs` — `And`, `MatchFirst`, `Or` (sequence, first-match, longest-match)
- `repetition.rs` — `ZeroOrMore`, `OneOrMore`, `Optional`, `Exactly`
- `structure.rs` — `Group`, `Suppress` (result nesting/filtering)
- `forward.rs` — `Forward` (placeholder for recursive grammars)

### Performance layers (`src/`)
Multiple optimization tiers in separate modules, each building on the last:
- `compiler.rs` / `compiled_grammar.rs` — `CompiledGrammar`, `FastParser`, `FastScanner` for pre-compiled grammar state machines
- `batch.rs` — Basic batch parsing operations
- `ultra_batch.rs` — High-performance batch with optimized allocation
- `parallel_batch.rs` — Rayon-based parallel batch processing with SIMD techniques
- `numpy_batch.rs` — Aggregation operations (count-only, avoiding object creation)
- `file_batch.rs` — Memory-mapped file I/O processing via `memmap2`

### Python bindings (`src/lib.rs`)
All `Py*` wrapper classes (e.g. `PyLiteral`, `PyWord`, `PyAnd`) are defined here. Each wraps its Rust parser in `Arc<dyn ParserElement>` and exposes `parse_string()`, `search_string()`, `parse_batch()`, plus operator overloading (`__add__` for `And`, `__or__` for `MatchFirst`).

## Key Design Decisions

- **Zero-copy parsing**: Parsers operate on `&str` slices of the original input, avoiding allocation.
- **256-bit CharSet**: `chars.rs` uses a 4x64-bit array for O(1) ASCII character membership tests with branchless bit ops.
- **First-byte fast path**: Literal matching checks the first character before full string comparison.
- **Arc-wrapped trait objects**: Parsers are shared via `Arc<dyn ParserElement>` to enable composition.
- **Aggressive release profile**: LTO, single codegen unit, panic=abort, stripped symbols, opt-level=3.

## Tests

Tests are in `tests/` as Python pytest files (they test the compiled Python module, not Rust directly):
- `test_basic.py` — Core element functionality
- `test_combinators.py` — Combinator composition and operator overloading
- `test_arithmetic.py` — Arithmetic expression grammar benchmark
- `test_performance.py` — Performance regression checks against baselines

## Continuous Optimization Protocol

When running as part of an agent team, follow these rules:

1. **Never stop**: There is always more performance to extract. After each optimization, identify the next bottleneck.
2. **Measure everything**: No optimization is committed without before/after benchmark numbers.
3. **Track state**: Update `TEAM_PROGRESS.md` after every optimization cycle so the next session can resume.
4. **Push frequently**: After each validated optimization (tests pass + perf improved), commit and push.
5. **Revert on regression**: If any benchmark gets slower, revert immediately and try a different approach.
6. **Log failures**: Record failed approaches in `TEAM_PROGRESS.md` so they aren't retried.
7. **Pre-commit checks**: Always run `cargo fmt && cargo clippy --all -- -D warnings` before committing.
8. **Python compat**: Must work on Python 3.9-3.13. Never manipulate `ob_refcnt` directly — use `Py_INCREF` loops.
