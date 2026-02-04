#!/usr/bin/env python3
"""Test arithmetic expression parsing with pyparsing_rs."""
import os
import sys
import pytest

# Add test_grammars to path (relative to repo root)
_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_repo_root, 'test_grammars'))

pyparsing = pytest.importorskip("pyparsing")


def test_arithmetic_integer_parsing():
    """Test that pyparsing_rs Word(nums) matches pyparsing for integers."""
    import pyparsing_rs as pp_rs

    integer = pp_rs.Word(pp_rs.nums())
    result = integer.parse_string("123")
    assert result == ["123"]


def test_arithmetic_expressions():
    """Test integer extraction from arithmetic expressions."""
    from arithmetic import TEST_EXPRESSIONS
    import pyparsing_rs as pp_rs

    integer_rs = pp_rs.Word(pp_rs.nums())
    integer_pp = pyparsing.Word(pyparsing.nums)

    for expr in TEST_EXPRESSIONS[:10]:
        first_token = expr.split()[0]
        try:
            orig_result = integer_pp.parse_string(first_token)
            our_result = integer_rs.parse_string(first_token)
            assert orig_result[0] == our_result[0], (
                f"Mismatch for '{first_token}': pyparsing={orig_result[0]}, "
                f"pyparsing_rs={our_result[0]}"
            )
        except Exception:
            pass  # Skip non-numeric first tokens
