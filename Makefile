# Makefile for pyparsing-rs development

.PHONY: help build dev test bench clean format lint check

help:
	@echo "Available targets:"
	@echo "  build    - Build the Python package (release mode)"
	@echo "  dev      - Build and install in development mode"
	@echo "  test     - Run all tests"
	@echo "  bench    - Run performance benchmarks"
	@echo "  clean    - Clean build artifacts"
	@echo "  format   - Format Rust code"
	@echo "  lint     - Run clippy linter"
	@echo "  check    - Run format check + lint + tests"

build:
	maturin build --release

dev:
	maturin develop --release

test: dev
	pytest tests/ -v

bench: dev
	python tests/test_performance.py

clean:
	cargo clean
	rm -rf target build dist
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

format:
	cargo fmt --all

lint:
	cargo clippy --all -- -D warnings

check: format
	cargo fmt --all -- --check
	cargo clippy --all -- -D warnings
	cargo test --all
	$(MAKE) test
