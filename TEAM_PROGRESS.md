# Agent Team Progress Tracker

## Status: RUNNING
## Started: 2026-02-06
## Iteration: 13

### Optimization Log

#### Cycle 12 (2026-02-06)
**Critical bug fix:** All 4 fixed-size (32-slot) hash tables had infinite loop bugs when >32 unique items were encountered:
- `hash_cache_batch_count`: added probe limit, bypasses cache when full
- Word `parse_batch` fallback: replaced with FxHashMap dedup (unlimited unique inputs)
- Regex `parse_batch` fallback: replaced with FxHashMap dedup (unlimited unique inputs)
- PyAnd `parse_batch` fallback: added probe limit + filled counter

**Other changes:**
- Word `search_string` fallback: two-pass approach with FxHashMap dedup + `count_words_branchless`
- Fixed type mismatch: `search_string` now uses `[u8; 256]` for is_init/is_body arrays

#### Cycle 11 (2026-02-06)
- Literal `parse_batch` uniform path: `PySequence_Repeat` replaces `bulk_incref` loop
- `generic_parse_batch` uniform path: same `PySequence_Repeat` optimization
- PyAnd: added `search_string` and `search_string_count` methods

#### Cycle 10 (2026-02-06)
- PyKeyword `parse_string`: `try_match_at` instead of `generic_parse_string`
- Word `search_string` cycle path: `PySequence_Repeat` for no-remainder case
- PyLiteral `search_string_count`: refactored to use `detect_text_period`

### Failed Approaches (do NOT retry)
- `generic_parse_string` with `try_match_at`: Breaks ZeroOrMore/OneOrMore multi-token semantics
- `generic_parse_batch` with `try_match_at`: Same multi-token issue
- `PyMatchFirst::parse_string` with `try_match_at`: Breaks when wrapping And (multi-token)

### Current Benchmark Results (venv Python, high variance at sub-ms)
| Benchmark | Speedup Range |
|---|---|
| literal_batch_parse | 700-900x |
| word_batch_parse | 2000-2400x |
| regex_batch_parse | 2000-2500x |
| word_search_string | 800-2000x |
| complex_batch_parse | 2700-2900x |
| literal_search_string | 14000-16000x |
| literal_batch_count | 5000-7500x |
| word_search_count | 140K-300Kx |
| literal_search_count | 370K-420Kx |
| complex_batch_count | 1.3M-1.4Mx |

**Note:** Count/search benchmarks have extreme speedups because Rust time is <0.01ms (at measurement floor).

### Performance Floor Analysis
We are approaching CPython overhead limits:
- **Python function call overhead**: ~100ns per call via PyO3 (irreducible)
- **CPython list creation**: ~5ns per item for PySequence_Repeat (C-level)
- **Literal parse_batch (10K uniform)**: 49.5μs → 4.9ns/item (near PySequence_Repeat floor)
- **parse_batch_count (10K uniform)**: 5.9μs → 0.6ns/item (near L1 cache access floor)

### Next Targets
- Profile non-benchmark scenarios (large grammars, recursive parsers)
- Consider PGO (profile-guided optimization) for the Rust compilation
- Explore `PyTuple` output for immutable results (faster creation than lists)
- Add more Python API compatibility methods (transform_string, run_tests)
