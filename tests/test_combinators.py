#!/usr/bin/env python3
"""Test combinators for pyparsing_rs."""
import pytest
import pyparsing_rs as pp

class TestAnd:
    def test_and_sequence(self):
        lit1 = pp.Literal("hello")
        lit2 = pp.Literal(" world")
        combined = lit1 + lit2
        result = combined.parse_string("hello world")
        assert result == ["hello", " world"]
    
    def test_and_mismatch(self):
        lit1 = pp.Literal("hello")
        lit2 = pp.Literal(" world")
        combined = lit1 + lit2
        with pytest.raises(ValueError):
            combined.parse_string("hello there")

class TestMatchFirst:
    def test_match_first_first_wins(self):
        lit1 = pp.Literal("hello")
        lit2 = pp.Literal("goodbye")
        combined = lit1 | lit2
        result = combined.parse_string("hello")
        assert result == ["hello"]
    
    def test_match_first_second(self):
        lit1 = pp.Literal("hello")
        lit2 = pp.Literal("goodbye")
        combined = lit1 | lit2
        result = combined.parse_string("goodbye")
        assert result == ["goodbye"]

class TestZeroOrMore:
    def test_zero_or_more_multiple(self):
        lit = pp.Literal("a")
        many = pp.ZeroOrMore(lit)
        result = many.parse_string("aaaa")
        assert result == ["a", "a", "a", "a"]
    
    def test_zero_or_more_zero(self):
        lit = pp.Literal("a")
        many = pp.ZeroOrMore(lit)
        result = many.parse_string("bbbb")
        assert result == []

class TestOneOrMore:
    def test_one_or_more_multiple(self):
        lit = pp.Literal("a")
        many = pp.OneOrMore(lit)
        result = many.parse_string("aaaa")
        assert result == ["a", "a", "a", "a"]
    
    def test_one_or_more_zero_fails(self):
        lit = pp.Literal("a")
        many = pp.OneOrMore(lit)
        with pytest.raises(ValueError):
            many.parse_string("bbbb")

class TestOptional:
    def test_optional_present(self):
        lit = pp.Literal("a")
        opt = pp.Optional(lit)
        result = opt.parse_string("a")
        assert result == ["a"]
    
    def test_optional_absent(self):
        lit = pp.Literal("a")
        opt = pp.Optional(lit)
        result = opt.parse_string("b")
        assert result == []

class TestSuppress:
    def test_suppress_no_output(self):
        lit = pp.Literal("hello")
        sup = pp.Suppress(lit)
        result = sup.parse_string("hello")
        assert result == []

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
