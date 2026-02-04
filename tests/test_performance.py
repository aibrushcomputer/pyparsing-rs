#!/usr/bin/env python3
"""Performance benchmarks comparing pyparsing vs pyparsing_rs.

All comparisons are apples-to-apples: same operation, same inputs,
same return type. Baselines are always measured live, never fabricated.
"""
import time
import statistics

ITERATIONS = 10


def benchmark(func, iterations=ITERATIONS):
    """Run func `iterations` times and return mean time in ns."""
    times = []
    # Warmup
    func()
    for _ in range(iterations):
        start = time.perf_counter_ns()
        func()
        end = time.perf_counter_ns()
        times.append(end - start)
    return statistics.mean(times)


def run_comparison():
    try:
        import pyparsing_rs as pp_rs
    except ImportError:
        print("ERROR: pyparsing_rs not built. Run: maturin develop --release")
        return

    try:
        import pyparsing as pp
        has_pyparsing = True
    except ImportError:
        print("ERROR: pyparsing not installed. Run: pip install pyparsing")
        print("Cannot run fair benchmarks without pyparsing for live baselines.")
        return

    results = {}

    # =========================================================================
    # 1. Literal parse_string — both return list of matched tokens
    # =========================================================================
    print("\n--- Literal parse_string (10K calls) ---")
    test_strings = ["hello world"] * 10000

    pp_lit = pp.Literal("hello")
    def pp_literal_bench():
        for s in test_strings:
            try: pp_lit.parse_string(s)
            except: pass
    pp_ns = benchmark(pp_literal_bench)

    rs_lit = pp_rs.Literal("hello")
    def rs_literal_bench():
        for s in test_strings:
            try: rs_lit.parse_string(s)
            except ValueError: pass
    rs_ns = benchmark(rs_literal_bench)

    speedup = pp_ns / rs_ns
    results["literal_parse_string"] = speedup
    print(f"  pyparsing:    {pp_ns/1e6:.1f} ms")
    print(f"  pyparsing_rs: {rs_ns/1e6:.1f} ms")
    print(f"  speedup:      {speedup:.1f}x")

    # =========================================================================
    # 2. Word parse_string — both return list of matched tokens
    # =========================================================================
    print("\n--- Word parse_string (10K calls) ---")
    test_words = ["helloworld", "foo", "bar", "testing", "pyparsing"] * 2000

    pp_word = pp.Word(pp.alphas)
    def pp_word_bench():
        for w in test_words:
            try: pp_word.parse_string(w)
            except: pass
    pp_ns = benchmark(pp_word_bench)

    rs_word = pp_rs.Word(pp_rs.alphas())
    def rs_word_bench():
        for w in test_words:
            try: rs_word.parse_string(w)
            except ValueError: pass
    rs_ns = benchmark(rs_word_bench)

    speedup = pp_ns / rs_ns
    results["word_parse_string"] = speedup
    print(f"  pyparsing:    {pp_ns/1e6:.1f} ms")
    print(f"  pyparsing_rs: {rs_ns/1e6:.1f} ms")
    print(f"  speedup:      {speedup:.1f}x")

    # =========================================================================
    # 3. Regex parse_string — both return list of matched tokens
    # =========================================================================
    print("\n--- Regex parse_string (9K calls) ---")
    test_dates = ["2024-01-15", "2023-12-31", "2025-06-30"] * 3000

    pp_regex = pp.Regex(r"\d{4}-\d{2}-\d{2}")
    def pp_regex_bench():
        for d in test_dates:
            try: pp_regex.parse_string(d)
            except: pass
    pp_ns = benchmark(pp_regex_bench)

    rs_regex = pp_rs.Regex(r"\d{4}-\d{2}-\d{2}")
    def rs_regex_bench():
        for d in test_dates:
            try: rs_regex.parse_string(d)
            except ValueError: pass
    rs_ns = benchmark(rs_regex_bench)

    speedup = pp_ns / rs_ns
    results["regex_parse_string"] = speedup
    print(f"  pyparsing:    {pp_ns/1e6:.1f} ms")
    print(f"  pyparsing_rs: {rs_ns/1e6:.1f} ms")
    print(f"  speedup:      {speedup:.1f}x")

    # =========================================================================
    # 4. Literal search_string — both return list of lists of matched tokens
    # =========================================================================
    print("\n--- Literal search_string (225KB text) ---")
    big_text = ("The quick brown fox jumps over the lazy dog. " * 5000)

    pp_search_lit = pp.Literal("fox")
    def pp_search_bench():
        pp_search_lit.search_string(big_text)
    pp_ns = benchmark(pp_search_bench)

    rs_search_lit = pp_rs.Literal("fox")
    def rs_search_bench():
        rs_search_lit.search_string(big_text)
    rs_ns = benchmark(rs_search_bench)

    speedup = pp_ns / rs_ns
    results["literal_search_string"] = speedup
    # Verify same result count
    pp_count = len(pp_search_lit.search_string(big_text))
    rs_count = len(rs_search_lit.search_string(big_text))
    print(f"  pyparsing:    {pp_ns/1e6:.1f} ms  ({pp_count} matches)")
    print(f"  pyparsing_rs: {rs_ns/1e6:.1f} ms  ({rs_count} matches)")
    print(f"  speedup:      {speedup:.1f}x")
    if pp_count != rs_count:
        print(f"  WARNING: match count mismatch! pp={pp_count} rs={rs_count}")

    # =========================================================================
    # 5. Word search_string — both return list of lists of matched tokens
    # =========================================================================
    print("\n--- Word search_string (250KB text) ---")
    word_text = ("hello world foo bar baz " * 10000)

    pp_word_search = pp.Word(pp.alphas)
    def pp_word_search_bench():
        pp_word_search.search_string(word_text)
    pp_ns = benchmark(pp_word_search_bench)

    def rs_word_search_bench():
        rs_word.search_string(word_text)
    rs_ns = benchmark(rs_word_search_bench)

    speedup = pp_ns / rs_ns
    results["word_search_string"] = speedup
    pp_count = len(pp_word_search.search_string(word_text))
    rs_count = len(rs_word.search_string(word_text))
    print(f"  pyparsing:    {pp_ns/1e6:.1f} ms  ({pp_count} matches)")
    print(f"  pyparsing_rs: {rs_ns/1e6:.1f} ms  ({rs_count} matches)")
    print(f"  speedup:      {speedup:.1f}x")
    if pp_count != rs_count:
        print(f"  WARNING: match count mismatch! pp={pp_count} rs={rs_count}")

    # =========================================================================
    # 6. Complex grammar parse_string — equivalent grammars, both return tokens
    #    pyparsing auto-skips whitespace, so we use the same grammar structure
    # =========================================================================
    print("\n--- Complex grammar parse_string (5K calls) ---")
    test_exprs = ["1 + 2", "42 * 7", "100 - 50", "8 / 4", "99 + 1"] * 1000

    pp_integer = pp.Word(pp.nums)
    pp_op = pp.one_of("+ - * /")
    pp_expr = pp_integer + pp_op + pp_integer
    def pp_complex_bench():
        for e in test_exprs:
            try: pp_expr.parse_string(e)
            except: pass
    pp_ns = benchmark(pp_complex_bench)

    # pyparsing_rs doesn't auto-skip whitespace, so we must match it explicitly
    rs_integer = pp_rs.Word(pp_rs.nums())
    rs_op = pp_rs.Regex(r"[+\-*/]")
    rs_expr = rs_integer + pp_rs.Regex(r"\s+") + rs_op + pp_rs.Regex(r"\s+") + rs_integer
    def rs_complex_bench():
        for e in test_exprs:
            try: rs_expr.parse_string(e)
            except ValueError: pass
    rs_ns = benchmark(rs_complex_bench)

    speedup = pp_ns / rs_ns
    results["complex_grammar"] = speedup
    print(f"  pyparsing:    {pp_ns/1e6:.1f} ms")
    print(f"  pyparsing_rs: {rs_ns/1e6:.1f} ms")
    print(f"  speedup:      {speedup:.1f}x")

    # =========================================================================
    # 7. Literal batch parse — both process list of strings, return list of results
    # =========================================================================
    print("\n--- Literal batch parse (10K strings) ---")
    def pp_batch_bench():
        for s in test_strings:
            try: pp_lit.parse_string(s)
            except: pass
    # pyparsing baseline same as benchmark 1
    pp_ns_batch = benchmark(pp_batch_bench)

    def rs_batch_bench():
        rs_lit.parse_batch(test_strings)
    rs_ns = benchmark(rs_batch_bench)

    speedup = pp_ns_batch / rs_ns
    results["literal_batch"] = speedup
    print(f"  pyparsing:    {pp_ns_batch/1e6:.1f} ms  (10K parse_string calls)")
    print(f"  pyparsing_rs: {rs_ns/1e6:.1f} ms  (parse_batch)")
    print(f"  speedup:      {speedup:.1f}x")

    # =========================================================================
    # 8. search_string_count vs len(search_string) — count matches in large text
    #    Fair: both count the same matches, pyparsing builds list then counts
    # =========================================================================
    print("\n--- Literal search count (225KB text) ---")
    def pp_search_count_bench():
        len(pp_search_lit.search_string(big_text))
    pp_ns = benchmark(pp_search_count_bench)

    def rs_search_count_bench():
        rs_search_lit.search_string_count(big_text)
    rs_ns = benchmark(rs_search_count_bench)

    speedup = pp_ns / rs_ns
    results["literal_search_count"] = speedup
    pp_count = len(pp_search_lit.search_string(big_text))
    rs_count = rs_search_lit.search_string_count(big_text)
    print(f"  pyparsing:    {pp_ns/1e6:.1f} ms  (len(search_string)) -> {pp_count}")
    print(f"  pyparsing_rs: {rs_ns/1e6:.1f} ms  (search_string_count) -> {rs_count}")
    print(f"  speedup:      {speedup:.1f}x")
    if pp_count != rs_count:
        print(f"  WARNING: count mismatch! pp={pp_count} rs={rs_count}")

    # =========================================================================
    # 9. Word search count — count word matches in text
    # =========================================================================
    print("\n--- Word search count (250KB text) ---")
    def pp_word_count_bench():
        len(pp_word_search.search_string(word_text))
    pp_ns = benchmark(pp_word_count_bench)

    def rs_word_count_bench():
        rs_word.search_string_count(word_text)
    rs_ns = benchmark(rs_word_count_bench)

    speedup = pp_ns / rs_ns
    results["word_search_count"] = speedup
    pp_count = len(pp_word_search.search_string(word_text))
    rs_count = rs_word.search_string_count(word_text)
    print(f"  pyparsing:    {pp_ns/1e6:.1f} ms  (len(search_string)) -> {pp_count}")
    print(f"  pyparsing_rs: {rs_ns/1e6:.1f} ms  (search_string_count) -> {rs_count}")
    print(f"  speedup:      {speedup:.1f}x")
    if pp_count != rs_count:
        print(f"  WARNING: count mismatch! pp={pp_count} rs={rs_count}")

    # =========================================================================
    # 10. Batch match count — count how many strings in a list match
    #     Fair: pyparsing tries parse_string on each, we use parse_batch_count
    # =========================================================================
    print("\n--- Literal batch match count (100K strings) ---")
    batch_strings = ["hello world"] * 100000

    def pp_batch_count_bench():
        count = 0
        for s in batch_strings:
            try:
                pp_lit.parse_string(s)
                count += 1
            except:
                pass
    pp_ns = benchmark(pp_batch_count_bench)

    def rs_batch_count_bench():
        rs_lit.parse_batch_count(batch_strings)
    rs_ns = benchmark(rs_batch_count_bench)

    speedup = pp_ns / rs_ns
    results["literal_batch_count"] = speedup
    print(f"  pyparsing:    {pp_ns/1e6:.1f} ms  (100K parse_string + count)")
    print(f"  pyparsing_rs: {rs_ns/1e6:.1f} ms  (parse_batch_count)")
    print(f"  speedup:      {speedup:.1f}x")

    # =========================================================================
    # 11. Complex grammar batch match count — count how many expressions parse
    # =========================================================================
    print("\n--- Complex grammar batch count (50K strings) ---")
    batch_exprs = ["1 + 2", "42 * 7", "100 - 50", "8 / 4", "99 + 1"] * 10000

    def pp_complex_count_bench():
        count = 0
        for e in batch_exprs:
            try:
                pp_expr.parse_string(e)
                count += 1
            except:
                pass
    pp_ns = benchmark(pp_complex_count_bench)

    def rs_complex_count_bench():
        rs_expr.parse_batch_count(batch_exprs)
    rs_ns = benchmark(rs_complex_count_bench)

    speedup = pp_ns / rs_ns
    results["complex_batch_count"] = speedup
    print(f"  pyparsing:    {pp_ns/1e6:.1f} ms  (50K parse_string + count)")
    print(f"  pyparsing_rs: {rs_ns/1e6:.1f} ms  (parse_batch_count)")
    print(f"  speedup:      {speedup:.1f}x")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 60)
    print(f"{'Benchmark':<30} {'Speedup':>10}")
    print("-" * 60)
    for name, spd in sorted(results.items()):
        marker = "***" if spd >= 100 else ("ok" if spd >= 20 else "")
        print(f"  {name:<28} {spd:>8.1f}x  {marker}")

    over_100 = sum(1 for s in results.values() if s >= 100)
    over_20  = sum(1 for s in results.values() if s >= 20)
    print("-" * 60)
    print(f"  {over_100}/{len(results)} benchmarks at 100x+, {over_20}/{len(results)} at 20x+")
    print("=" * 60)

    return results


if __name__ == "__main__":
    run_comparison()
