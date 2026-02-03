#!/usr/bin/env python3
"""Performance benchmarks comparing pyparsing vs pyparsing_rs."""
import time
import json
import statistics
from pathlib import Path

BASELINE_FILE = Path("/home/aibrush/pyparsing-rs/baseline_results.json")
RESULTS_FILE = Path("/home/aibrush/pyparsing-rs/performance_results.json")
ITERATIONS = 10

def load_baseline():
    if not BASELINE_FILE.exists():
        print(f"ERROR: Baseline not found at {BASELINE_FILE}")
        print("Run: python baseline_benchmark.py")
        return None
    with open(BASELINE_FILE) as f:
        return json.load(f)

def benchmark(func, iterations=ITERATIONS):
    times = []
    for _ in range(iterations):
        start = time.perf_counter_ns()
        func()
        end = time.perf_counter_ns()
        times.append(end - start)
    return {"mean_ns": statistics.mean(times)}

def run_comparison():
    baseline = load_baseline()
    if baseline is None:
        return
    
    try:
        import pyparsing_rs as pp_rs
    except ImportError:
        print("ERROR: pyparsing_rs not built. Run: maturin develop --release")
        return
    
    results = {}
    
    # Simple literal benchmark - SINGLE
    print("\nBenchmarking Literal matching (single)...")
    lit = pp_rs.Literal("hello")
    test_strings = ["hello world"] * 10000
    
    def literal_bench():
        for s in test_strings:
            try:
                lit.parse_string(s)
            except ValueError:
                pass
    
    rust_result = benchmark(literal_bench)
    orig = baseline.get("simple_literal", {})
    
    if orig:
        speedup = orig["mean_ns"] / rust_result["mean_ns"]
        results["literal_single"] = {"speedup": speedup, "target_met": speedup >= 50}
        print(f"  Literal (single): {speedup:.1f}x speedup {'✓' if speedup >= 50 else '✗'}")
    
    # Simple literal benchmark - BATCH
    print("Benchmarking Literal matching (batch)...")
    def literal_batch_bench():
        lit.parse_batch(test_strings)
    
    rust_result_batch = benchmark(literal_batch_bench)
    
    if orig:
        speedup_batch = orig["mean_ns"] / rust_result_batch["mean_ns"]
        results["literal_batch"] = {"speedup": speedup_batch, "target_met": speedup_batch >= 50}
        print(f"  Literal (batch): {speedup_batch:.1f}x speedup {'✓' if speedup_batch >= 50 else '✗'}")
    
    # Word benchmark - SINGLE
    print("Benchmarking Word matching (single)...")
    word = pp_rs.Word(pp_rs.alphas())
    test_words = ["helloworld", "foo", "bar", "testing", "pyparsing"] * 2000
    
    def word_bench():
        for w in test_words:
            try:
                word.parse_string(w)
            except ValueError:
                pass
    
    rust_result = benchmark(word_bench)
    orig = baseline.get("word_match", {})
    
    if orig:
        speedup = orig["mean_ns"] / rust_result["mean_ns"]
        results["word_single"] = {"speedup": speedup, "target_met": speedup >= 50}
        print(f"  Word (single): {speedup:.1f}x speedup {'✓' if speedup >= 50 else '✗'}")
    
    # Word benchmark - BATCH
    print("Benchmarking Word matching (batch)...")
    def word_batch_bench():
        word.parse_batch(test_words)
    
    rust_result_batch = benchmark(word_batch_bench)
    
    if orig:
        speedup_batch = orig["mean_ns"] / rust_result_batch["mean_ns"]
        results["word_batch"] = {"speedup": speedup_batch, "target_met": speedup_batch >= 50}
        print(f"  Word (batch): {speedup_batch:.1f}x speedup {'✓' if speedup_batch >= 50 else '✗'}")
    
    # Regex benchmark
    print("Benchmarking Regex matching...")
    regex = pp_rs.Regex(r"\d{4}-\d{2}-\d{2}")
    test_dates = ["2024-01-15", "2023-12-31", "2025-06-30"] * 3000
    
    def regex_bench():
        for d in test_dates:
            try:
                regex.parse_string(d)
            except ValueError:
                pass
    
    rust_result = benchmark(regex_bench)
    orig = baseline.get("regex_match", {})
    
    if orig:
        speedup = orig["mean_ns"] / rust_result["mean_ns"]
        results["regex_match"] = {"speedup": speedup, "target_met": speedup >= 50}
        print(f"  Regex: {speedup:.1f}x speedup {'✓' if speedup >= 50 else '✗'}")
    
    # Summary
    print("\n" + "="*50)
    all_met = all(r.get("target_met", False) for r in results.values())
    if all_met:
        print("🎉 ALL TARGETS MET! 50x+ improvement achieved!")
    else:
        not_met = [k for k, v in results.items() if not v.get("target_met")]
        print(f"Targets not yet met: {not_met}")
    
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)
    
    return results

if __name__ == "__main__":
    run_comparison()
