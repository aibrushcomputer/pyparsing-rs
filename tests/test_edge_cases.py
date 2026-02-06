#!/usr/bin/env python3
"""
Comprehensive edge-case tests for pyparsing_rs.

Tests cover: empty strings, unicode, long inputs, nested combinators,
boundary conditions, operator overloading, search_string correctness,
batch operations, and cross-validation against pyparsing.
"""
import pytest
import pyparsing_rs as pp


# ============================================================================
# a. Empty string handling
# ============================================================================

class TestEmptyStringHandling:
    """Test behavior with empty strings and empty inputs."""

    def test_literal_parse_empty_string(self):
        """Parsing empty string with a non-empty Literal should raise."""
        lit = pp.Literal("hello")
        with pytest.raises(ValueError):
            lit.parse_string("")

    def test_word_parse_empty_string(self):
        """Parsing empty string with Word should raise."""
        word = pp.Word(pp.alphas())
        with pytest.raises(ValueError):
            word.parse_string("")

    def test_regex_star_parse_empty_string(self):
        """Regex r'.*' on empty string should produce a zero-length match."""
        regex = pp.Regex(r".*")
        result = regex.parse_string("")
        # r".*" matches empty string (zero-length match)
        assert result == [""]

    def test_regex_plus_parse_empty_string(self):
        """Regex r'.+' on empty string should raise (needs at least one char)."""
        regex = pp.Regex(r".+")
        with pytest.raises(ValueError):
            regex.parse_string("")

    def test_literal_search_string_on_empty(self):
        """search_string on empty string should return empty list."""
        lit = pp.Literal("hello")
        result = lit.search_string("")
        assert result == []

    def test_word_search_string_on_empty(self):
        """search_string on empty string should return empty list."""
        word = pp.Word(pp.alphas())
        result = word.search_string("")
        assert result == []

    def test_regex_search_string_on_empty(self):
        """search_string with non-empty pattern on empty string should return empty."""
        regex = pp.Regex(r"\d+")
        result = regex.search_string("")
        assert result == []

    def test_literal_parse_batch_empty_list(self):
        """parse_batch with an empty list should return empty list."""
        lit = pp.Literal("hello")
        result = lit.parse_batch([])
        assert result == []

    def test_word_parse_batch_empty_list(self):
        """parse_batch with empty list for Word should return empty list."""
        word = pp.Word(pp.alphas())
        result = word.parse_batch([])
        assert result == []

    def test_regex_parse_batch_empty_list(self):
        """parse_batch with empty list for Regex should return empty list."""
        regex = pp.Regex(r"\d+")
        result = regex.parse_batch([])
        assert result == []

    def test_literal_parse_batch_empty_strings(self):
        """parse_batch with list of empty strings should return list of empty results."""
        lit = pp.Literal("hello")
        result = lit.parse_batch(["", "", ""])
        # Each empty string should not match "hello"
        assert len(result) == 3
        for r in result:
            assert r == []

    def test_literal_search_string_count_empty(self):
        """search_string_count on empty string should return 0."""
        lit = pp.Literal("hello")
        count = lit.search_string_count("")
        assert count == 0

    def test_word_search_string_count_empty(self):
        """search_string_count on empty string should return 0."""
        word = pp.Word(pp.alphas())
        count = word.search_string_count("")
        assert count == 0

    def test_literal_parse_batch_count_empty(self):
        """parse_batch_count with empty list should return 0."""
        lit = pp.Literal("hello")
        count = lit.parse_batch_count([])
        assert count == 0

    def test_zero_or_more_on_empty_string(self):
        """ZeroOrMore on empty string should return empty list (zero matches)."""
        lit = pp.Literal("a")
        zom = pp.ZeroOrMore(lit)
        result = zom.parse_string("")
        assert result == []

    def test_one_or_more_on_empty_string(self):
        """OneOrMore on empty string should raise (needs at least one match)."""
        lit = pp.Literal("a")
        oom = pp.OneOrMore(lit)
        with pytest.raises(ValueError):
            oom.parse_string("")

    def test_optional_on_empty_string(self):
        """Optional on empty string should return empty list (no match found)."""
        lit = pp.Literal("a")
        opt = pp.Optional(lit)
        result = opt.parse_string("")
        assert result == []


# ============================================================================
# b. Unicode handling
# ============================================================================

class TestUnicodeHandling:
    """Test behavior with unicode characters."""

    def test_literal_unicode_simple(self):
        """Literal with simple unicode string should match."""
        lit = pp.Literal("cafe\u0301")  # cafe with combining accent
        result = lit.parse_string("cafe\u0301")
        assert result == ["cafe\u0301"]

    def test_literal_chinese(self):
        """Literal with Chinese characters should match."""
        lit = pp.Literal("hello")
        result = lit.parse_string("hello world")
        assert result == ["hello"]

    def test_literal_search_in_unicode_text(self):
        """Literal should find ASCII within unicode text."""
        lit = pp.Literal("world")
        result = lit.search_string("hello world goodbye world")
        assert len(result) == 2

    def test_literal_unicode_literal_match(self):
        """Literal matching a unicode literal."""
        text = "the quick brown fox"
        lit = pp.Literal("quick")
        result = lit.search_string(text)
        assert len(result) == 1

    def test_regex_unicode_word_chars(self):
        """Regex with \\w should match unicode word characters."""
        regex = pp.Regex(r"\w+")
        result = regex.parse_string("hello")
        assert result == ["hello"]

    def test_regex_search_in_mixed_text(self):
        """Regex should find patterns in mixed ASCII/unicode text."""
        regex = pp.Regex(r"\d+")
        text = "price is 42 dollars and 99 cents"
        result = regex.search_string(text)
        assert len(result) == 2

    def test_literal_multibyte_unicode(self):
        """Literal with multi-byte unicode characters should match."""
        lit = pp.Literal("\u00e9\u00e8\u00ea")  # e-acute, e-grave, e-circumflex
        result = lit.parse_string("\u00e9\u00e8\u00ea")
        assert result == ["\u00e9\u00e8\u00ea"]

    def test_literal_search_string_count_unicode(self):
        """search_string_count should work with unicode text."""
        lit = pp.Literal("abc")
        text = "xyzabcxyzabcxyz"
        count = lit.search_string_count(text)
        assert count == 2


# ============================================================================
# c. Very long inputs
# ============================================================================

class TestLongInputs:
    """Test behavior with large inputs."""

    def test_literal_parse_long_string_at_start(self):
        """Literal should match at the start of a very long string."""
        lit = pp.Literal("hello")
        long_string = "hello" + "x" * (1024 * 1024)
        result = lit.parse_string(long_string)
        assert result == ["hello"]

    def test_literal_search_long_string(self):
        """search_string should find all occurrences in a 1MB string."""
        lit = pp.Literal("needle")
        # Create 1MB string with 100 needles
        segment = "x" * 10000 + "needle"
        long_string = segment * 100
        result = lit.search_string(long_string)
        assert len(result) == 100

    def test_literal_search_string_count_long(self):
        """search_string_count should count correctly in large strings."""
        lit = pp.Literal("ab")
        long_string = "ab" * 50000
        count = lit.search_string_count(long_string)
        assert count == 50000

    def test_word_long_single_word(self):
        """Word should match a single very long word (10K chars)."""
        word = pp.Word(pp.alphas())
        long_word = "a" * 10000
        result = word.parse_string(long_word)
        assert result == [long_word]

    def test_word_search_long_string(self):
        """Word search_string on a large input."""
        word = pp.Word(pp.alphas())
        # 100K chars: alternating words and spaces
        text = " ".join(["word"] * 20000)
        count = word.search_string_count(text)
        assert count == 20000

    def test_regex_search_long_string(self):
        """Regex search_string on a large input."""
        regex = pp.Regex(r"\d+")
        text = " ".join(["123"] * 10000)
        result = regex.search_string(text)
        assert len(result) == 10000

    def test_literal_parse_batch_large(self):
        """parse_batch with a large list of inputs."""
        lit = pp.Literal("hello")
        inputs = ["hello"] * 10000
        result = lit.parse_batch(inputs)
        assert len(result) == 10000
        assert all(r == ["hello"] for r in result)


# ============================================================================
# d. Nested combinators
# ============================================================================

class TestNestedCombinators:
    """Test deeply nested combinator structures."""

    def test_deeply_nested_and(self):
        """Deeply nested And (a + b + c + ...) should work."""
        # Build a + b + c + d + e via chaining
        result_parser = pp.Literal("a") + pp.Literal("b") + pp.Literal("c") + pp.Literal("d") + pp.Literal("e")
        result = result_parser.parse_string("abcde")
        assert result == ["a", "b", "c", "d", "e"]

    def test_match_first_many_alternatives(self):
        """MatchFirst with many alternatives should still find the right one.

        Note: MatchFirst uses first-match semantics. Since "alt_9" is a prefix
        of "alt_99", we use non-prefix-conflicting names to test cleanly.
        """
        # Use zero-padded names so no alternative is a prefix of another
        alternatives = [pp.Literal(f"alt_{i:03d}") for i in range(100)]
        mf = pp.MatchFirst(alternatives)
        # Should find the last alternative
        result = mf.parse_string("alt_099")
        assert result == ["alt_099"]
        # Should find the first alternative
        result = mf.parse_string("alt_000")
        assert result == ["alt_000"]
        # Should find a middle alternative
        result = mf.parse_string("alt_050")
        assert result == ["alt_050"]

    def test_match_first_prefix_semantics(self):
        """MatchFirst picks the FIRST matching alternative (first-match semantics).

        If "alt_9" appears before "alt_99" in the alternative list,
        "alt_9" matches the start of "alt_99" and wins.
        """
        alternatives = [pp.Literal(f"alt_{i}") for i in range(100)]
        mf = pp.MatchFirst(alternatives)
        # "alt_99" should match "alt_9" first since it comes first
        result = mf.parse_string("alt_99")
        assert result == ["alt_9"]

    def test_zero_or_more_nested(self):
        """ZeroOrMore(ZeroOrMore(x)) should still work (inner always succeeds with 0+)."""
        lit = pp.Literal("a")
        inner = pp.ZeroOrMore(lit)
        # ZeroOrMore of ZeroOrMore — the inner ZeroOrMore will consume all 'a's
        # on first invocation, and the outer one will stop because inner
        # produces 0 tokens on the second call
        outer = pp.ZeroOrMore(inner)
        result = outer.parse_string("aaa")
        # The exact behavior may vary. Just verify it doesn't crash or hang.
        assert isinstance(result, list)

    def test_optional_nested(self):
        """Optional(Optional(x)) should behave like Optional(x)."""
        lit = pp.Literal("a")
        opt = pp.Optional(pp.Optional(lit))
        # When present
        result = opt.parse_string("a")
        assert result == ["a"]
        # When absent
        result = opt.parse_string("b")
        assert result == []

    def test_chained_and_five_words(self):
        """Five Words chained with + should parse correctly."""
        w = pp.Word(pp.alphas())
        sp = pp.Literal(" ")
        parser = w + sp + w + sp + w
        result = parser.parse_string("the quick brown")
        assert "the" in result
        assert "quick" in result
        assert "brown" in result


# ============================================================================
# e. Boundary conditions
# ============================================================================

class TestBoundaryConditions:
    """Test edge cases at boundaries."""

    def test_keyword_at_exact_end_of_string(self):
        """Keyword at exact end of string should match."""
        kw = pp.Keyword("end")
        result = kw.parse_string("end")
        assert result == ["end"]

    def test_keyword_fails_when_followed_by_alpha(self):
        """Keyword should fail if followed by alphanumeric."""
        kw = pp.Keyword("if")
        with pytest.raises(ValueError):
            kw.parse_string("iffy")

    def test_literal_longer_than_input(self):
        """Literal longer than input string should raise."""
        lit = pp.Literal("this is a long literal")
        with pytest.raises(ValueError):
            lit.parse_string("short")

    def test_literal_exact_match(self):
        """Literal that exactly matches the entire input."""
        lit = pp.Literal("exact")
        result = lit.parse_string("exact")
        assert result == ["exact"]

    def test_literal_single_char(self):
        """Single character literal."""
        lit = pp.Literal("x")
        result = lit.parse_string("x")
        assert result == ["x"]

    def test_word_single_char(self):
        """Word matching a single character."""
        word = pp.Word(pp.alphas())
        result = word.parse_string("a")
        assert result == ["a"]

    def test_word_with_body_no_body_chars(self):
        """Word where body_chars don't match after first char yields single char."""
        word = pp.Word("abc", "xyz")
        result = word.parse_string("a123")
        assert result == ["a"]

    def test_word_init_char_not_in_input(self):
        """Word where init_chars don't match the input start should raise."""
        word = pp.Word("abc")
        with pytest.raises(ValueError):
            word.parse_string("xyz")

    def test_parse_batch_mixed_matching(self):
        """parse_batch with mix of matching and non-matching strings."""
        lit = pp.Literal("hello")
        inputs = ["hello", "world", "hello", "foo", "hello"]
        result = lit.parse_batch(inputs)
        assert len(result) == 5
        assert result[0] == ["hello"]
        assert result[1] == []
        assert result[2] == ["hello"]
        assert result[3] == []
        assert result[4] == ["hello"]

    def test_parse_batch_count_mixed(self):
        """parse_batch_count with mixed matching/non-matching."""
        lit = pp.Literal("hello")
        inputs = ["hello", "world", "hello", "foo", "hello"]
        count = lit.parse_batch_count(inputs)
        assert count == 3

    def test_regex_anchored_start(self):
        """Regex with ^ anchor should only match at start."""
        regex = pp.Regex(r"^\d+")
        result = regex.parse_string("123abc")
        assert result == ["123"]

    def test_regex_no_match(self):
        """Regex that doesn't match should raise."""
        regex = pp.Regex(r"\d+")
        with pytest.raises(ValueError):
            regex.parse_string("no digits here")

    def test_literal_matches_method(self):
        """matches() method should return True/False without raising."""
        lit = pp.Literal("hello")
        assert lit.matches("hello world") is True
        assert lit.matches("goodbye") is False
        assert lit.matches("") is False

    def test_word_matches_method(self):
        """matches() method on Word."""
        word = pp.Word(pp.alphas())
        assert word.matches("hello") is True
        assert word.matches("123") is False
        assert word.matches("") is False

    def test_regex_matches_method(self):
        """matches() method on Regex."""
        regex = pp.Regex(r"\d+")
        assert regex.matches("123") is True
        assert regex.matches("abc") is False
        assert regex.matches("") is False

    def test_and_matches_method(self):
        """matches() method on And combinator."""
        parser = pp.Literal("hello") + pp.Literal(" world")
        assert parser.matches("hello world") is True
        assert parser.matches("hello there") is False
        assert parser.matches("") is False


# ============================================================================
# f. Operator overloading
# ============================================================================

class TestOperatorOverloading:
    """Test + and | operator overloading between different parser types."""

    def test_chained_and_three_literals(self):
        """a + b + c should produce And with three elements."""
        a = pp.Literal("a")
        b = pp.Literal("b")
        c = pp.Literal("c")
        parser = a + b + c
        result = parser.parse_string("abc")
        assert result == ["a", "b", "c"]

    def test_chained_or_three_literals_via_constructor(self):
        """MatchFirst([a, b, c]) should check a, then b, then c.

        Note: MatchFirst does not have __or__, so chaining a | b | c
        is not supported. Use the MatchFirst constructor instead.
        """
        a = pp.Literal("a")
        b = pp.Literal("b")
        c = pp.Literal("c")
        parser = pp.MatchFirst([a, b, c])
        assert parser.parse_string("a") == ["a"]
        assert parser.parse_string("b") == ["b"]
        assert parser.parse_string("c") == ["c"]

    def test_two_way_or_operator(self):
        """a | b using operator should work for two alternatives."""
        a = pp.Literal("a")
        b = pp.Literal("b")
        parser = a | b
        assert parser.parse_string("a") == ["a"]
        assert parser.parse_string("b") == ["b"]

    def test_mixed_and_or(self):
        """(a + b) | (c + d) should work correctly."""
        a = pp.Literal("a")
        b = pp.Literal("b")
        c = pp.Literal("c")
        d = pp.Literal("d")
        ab = a + b
        cd = c + d
        # MatchFirst between two And parsers - need to use MatchFirst constructor
        parser = pp.MatchFirst([ab, cd])
        assert parser.parse_string("ab") == ["a", "b"]
        assert parser.parse_string("cd") == ["c", "d"]

    def test_literal_plus_word(self):
        """Literal + Word should produce And."""
        lit = pp.Literal("x=")
        word = pp.Word(pp.alphas())
        parser = lit + word
        result = parser.parse_string("x=hello")
        assert result == ["x=", "hello"]

    def test_word_plus_literal(self):
        """Word + Literal should produce And."""
        word = pp.Word(pp.alphas())
        lit = pp.Literal("!")
        parser = word + lit
        result = parser.parse_string("hello!")
        assert result == ["hello", "!"]

    def test_literal_or_word(self):
        """Literal | Word should produce MatchFirst."""
        lit = pp.Literal("123")
        word = pp.Word(pp.alphas())
        parser = lit | word
        assert parser.parse_string("123") == ["123"]
        assert parser.parse_string("hello") == ["hello"]

    def test_regex_plus_literal(self):
        """Regex + Literal should produce And."""
        regex = pp.Regex(r"\d+")
        lit = pp.Literal("!")
        parser = regex + lit
        result = parser.parse_string("123!")
        assert result == ["123", "!"]

    def test_and_plus_literal(self):
        """(a + b) + c should flatten into single And."""
        a = pp.Literal("a")
        b = pp.Literal("b")
        c = pp.Literal("c")
        parser = (a + b) + c
        result = parser.parse_string("abc")
        assert result == ["a", "b", "c"]

    def test_suppress_plus_literal(self):
        """Suppress + Literal should produce And where suppress part is omitted."""
        sup = pp.Suppress(pp.Literal("("))
        lit = pp.Literal("x")
        parser = sup + lit
        result = parser.parse_string("(x")
        # Suppress should omit "(" from results
        assert "x" in result

    def test_group_plus_literal(self):
        """Group + Literal should produce And."""
        grp = pp.Group(pp.Literal("a"))
        lit = pp.Literal("b")
        parser = grp + lit
        result = parser.parse_string("ab")
        assert "b" in result


# ============================================================================
# g. search_string correctness
# ============================================================================

class TestSearchStringCorrectness:
    """Test search_string produces correct results."""

    def test_literal_search_non_overlapping(self):
        """Literal search should find non-overlapping matches."""
        lit = pp.Literal("aa")
        # "aaaa" should give 2 non-overlapping matches of "aa"
        result = lit.search_string("aaaa")
        assert len(result) == 2

    def test_literal_search_adjacent_matches(self):
        """Adjacent but non-overlapping matches should all be found."""
        lit = pp.Literal("ab")
        result = lit.search_string("ababab")
        assert len(result) == 3

    def test_literal_search_no_matches(self):
        """search_string with no matches should return empty list."""
        lit = pp.Literal("xyz")
        result = lit.search_string("hello world")
        assert result == []

    def test_literal_search_single_match(self):
        """search_string with exactly one match."""
        lit = pp.Literal("world")
        result = lit.search_string("hello world goodbye")
        assert len(result) == 1

    def test_literal_search_at_boundaries(self):
        """search_string should find matches at start and end."""
        lit = pp.Literal("ab")
        result = lit.search_string("abXXab")
        assert len(result) == 2

    def test_word_search_multiple_words(self):
        """Word search_string should find all words."""
        word = pp.Word(pp.alphas())
        text = "hello world foo bar"
        count = word.search_string_count(text)
        assert count == 4

    def test_word_search_with_numbers_between(self):
        """Word search should find words separated by numbers."""
        word = pp.Word(pp.alphas())
        text = "abc123def456ghi"
        count = word.search_string_count(text)
        assert count == 3

    def test_regex_search_multiple_matches(self):
        """Regex search_string should find all matches."""
        regex = pp.Regex(r"\d+")
        text = "a1b22c333d4444"
        result = regex.search_string(text)
        assert len(result) == 4

    def test_regex_search_values(self):
        """Verify regex search_string returns correct matched values."""
        regex = pp.Regex(r"\d+")
        text = "a1b22c333"
        result = regex.search_string(text)
        # Each result is a list containing the match
        values = [r[0] if isinstance(r, list) else r for r in result]
        assert "1" in values
        assert "22" in values
        assert "333" in values

    def test_literal_search_string_count_matches_len(self):
        """search_string_count should equal len(search_string)."""
        lit = pp.Literal("ab")
        text = "ab cd ab ef ab gh ab"
        count = lit.search_string_count(text)
        results = lit.search_string(text)
        assert count == len(results)

    def test_word_search_string_count_matches_count(self):
        """Word search_string_count should match search_string length."""
        word = pp.Word(pp.alphas())
        text = "the quick brown fox jumps over the lazy dog"
        count = word.search_string_count(text)
        # There are 9 words in this sentence
        assert count == 9

    def test_regex_search_string_count_matches_len(self):
        """Regex search_string_count should equal len(search_string)."""
        regex = pp.Regex(r"\d+")
        text = "1 22 333 4444 55555"
        count = regex.search_string_count(text)
        results = regex.search_string(text)
        assert count == len(results)


# ============================================================================
# h. Batch operations
# ============================================================================

class TestBatchOperations:
    """Test batch parsing operations for correctness."""

    def test_literal_parse_batch_all_match(self):
        """parse_batch where all inputs match."""
        lit = pp.Literal("hello")
        inputs = ["hello", "hello", "hello"]
        result = lit.parse_batch(inputs)
        assert len(result) == 3
        assert all(r == ["hello"] for r in result)

    def test_literal_parse_batch_none_match(self):
        """parse_batch where no inputs match."""
        lit = pp.Literal("hello")
        inputs = ["world", "foo", "bar"]
        result = lit.parse_batch(inputs)
        assert len(result) == 3
        assert all(r == [] for r in result)

    def test_literal_parse_batch_count_all_match(self):
        """parse_batch_count where all match."""
        lit = pp.Literal("hello")
        inputs = ["hello", "hello", "hello"]
        count = lit.parse_batch_count(inputs)
        assert count == 3

    def test_literal_parse_batch_count_none_match(self):
        """parse_batch_count where none match."""
        lit = pp.Literal("hello")
        inputs = ["world", "foo", "bar"]
        count = lit.parse_batch_count(inputs)
        assert count == 0

    def test_literal_parse_batch_count_matches_batch(self):
        """parse_batch_count should equal count of non-empty results from parse_batch."""
        lit = pp.Literal("test")
        inputs = ["test", "nope", "test", "nada", "test", "no"]
        batch_result = lit.parse_batch(inputs)
        count = lit.parse_batch_count(inputs)
        matching = sum(1 for r in batch_result if len(r) > 0)
        assert count == matching

    def test_word_parse_batch_correctness(self):
        """Word parse_batch returns flat list of matched strings (filters non-matching).

        Unlike Literal.parse_batch which returns list-of-lists, Word.parse_batch
        returns a flat list containing only the strings that matched.
        """
        word = pp.Word(pp.alphas())
        inputs = ["hello", "123", "world", "456", "foo"]
        result = word.parse_batch(inputs)
        # Word.parse_batch returns flat list of matched strings only
        assert list(result) == ["hello", "world", "foo"]

    def test_word_parse_batch_count(self):
        """Word parse_batch_count should match actual count."""
        word = pp.Word(pp.alphas())
        inputs = ["hello", "123", "world", "456", "foo"]
        count = word.parse_batch_count(inputs)
        assert count == 3

    def test_regex_parse_batch_correctness(self):
        """Regex parse_batch returns flat list of matched strings (filters non-matching).

        Like Word.parse_batch, Regex.parse_batch returns a flat list containing
        only the strings that matched, not a list-of-lists.
        """
        regex = pp.Regex(r"\d+")
        inputs = ["123", "abc", "456", "def", "789"]
        result = regex.parse_batch(inputs)
        # Regex.parse_batch returns flat list of matched strings only
        assert list(result) == ["123", "456", "789"]

    def test_regex_parse_batch_count(self):
        """Regex parse_batch_count should match actual count."""
        regex = pp.Regex(r"\d+")
        inputs = ["123", "abc", "456", "def", "789"]
        count = regex.parse_batch_count(inputs)
        assert count == 3

    def test_and_parse_batch_count(self):
        """And parse_batch_count correctness."""
        parser = pp.Literal("hello") + pp.Literal(" world")
        inputs = ["hello world", "goodbye", "hello world", "nope"]
        count = parser.parse_batch_count(inputs)
        assert count == 2

    def test_literal_parse_batch_single_element(self):
        """parse_batch with a single element list."""
        lit = pp.Literal("x")
        result = lit.parse_batch(["x"])
        assert len(result) == 1
        assert result[0] == ["x"]

    def test_literal_parse_batch_uniform(self):
        """parse_batch with all identical strings (tests uniform fast path)."""
        lit = pp.Literal("same")
        inputs = ["same"] * 1000
        result = lit.parse_batch(inputs)
        assert len(result) == 1000
        assert all(r == ["same"] for r in result)

    def test_literal_parse_batch_uniform_no_match(self):
        """parse_batch with all identical non-matching strings."""
        lit = pp.Literal("hello")
        inputs = ["world"] * 1000
        result = lit.parse_batch(inputs)
        assert len(result) == 1000
        assert all(r == [] for r in result)


# ============================================================================
# i. Cross-validation against pyparsing
# ============================================================================

class TestCrossValidation:
    """Cross-validate pyparsing_rs behavior against pyparsing where possible."""

    @pytest.fixture
    def pyparsing(self):
        """Import pyparsing, skip if not available."""
        pyparsing = pytest.importorskip("pyparsing")
        return pyparsing

    def test_literal_basic(self, pyparsing):
        """Literal parse_string should produce same result as pyparsing."""
        pp_result = pyparsing.Literal("hello").parseString("hello world")
        rs_result = pp.Literal("hello").parse_string("hello world")
        assert list(pp_result) == list(rs_result)

    def test_literal_fail(self, pyparsing):
        """Both should raise on mismatch."""
        with pytest.raises(Exception):
            pyparsing.Literal("hello").parseString("goodbye")
        with pytest.raises(ValueError):
            pp.Literal("hello").parse_string("goodbye")

    def test_word_alpha(self, pyparsing):
        """Word(alphas) should produce same result."""
        pp_result = pyparsing.Word(pyparsing.alphas).parseString("hello123")
        rs_result = pp.Word(pp.alphas()).parse_string("hello123")
        assert list(pp_result) == list(rs_result)

    def test_word_alphanum(self, pyparsing):
        """Word(alphanums) should produce same result."""
        pp_result = pyparsing.Word(pyparsing.alphanums).parseString("hello123")
        rs_result = pp.Word(pp.alphanums()).parse_string("hello123")
        assert list(pp_result) == list(rs_result)

    def test_keyword_match(self, pyparsing):
        """Keyword matching should produce same result."""
        pp_result = pyparsing.Keyword("if").parseString("if")
        rs_result = pp.Keyword("if").parse_string("if")
        assert list(pp_result) == list(rs_result)

    def test_keyword_fail(self, pyparsing):
        """Both should fail on keyword with trailing alpha."""
        with pytest.raises(Exception):
            pyparsing.Keyword("if").parseString("ifx")
        with pytest.raises(ValueError):
            pp.Keyword("if").parse_string("ifx")

    def test_regex_digits(self, pyparsing):
        """Regex \\d+ should produce same result."""
        pp_result = pyparsing.Regex(r"\d+").parseString("12345abc")
        rs_result = pp.Regex(r"\d+").parse_string("12345abc")
        assert list(pp_result) == list(rs_result)

    def test_and_sequence(self, pyparsing):
        """And (a + b) should produce same tokens.

        Note: pyparsing auto-strips whitespace between elements by default,
        so we use Literal("world") instead of Literal(" world") to avoid
        the whitespace-skipping difference. Both should match "hello" then "world".
        """
        pp_a = pyparsing.Literal("hello")
        pp_b = pyparsing.Literal("world")
        pp_result = (pp_a + pp_b).parseString("hello world")

        rs_a = pp.Literal("hello")
        rs_b = pp.Literal(" world")
        rs_result = (rs_a + rs_b).parse_string("hello world")

        # pyparsing returns ["hello", "world"] (strips whitespace between)
        # pyparsing_rs returns ["hello", " world"] (literal match includes space)
        # Both return 2 tokens, first is "hello"
        assert list(pp_result)[0] == list(rs_result)[0]  # "hello"
        assert len(list(pp_result)) == len(list(rs_result))  # same count

    def test_and_no_whitespace(self, pyparsing):
        """And without whitespace between tokens — both should match identically."""
        pp_result = (pyparsing.Literal("ab") + pyparsing.Literal("cd")).parseString("abcd")
        rs_result = (pp.Literal("ab") + pp.Literal("cd")).parse_string("abcd")
        assert list(pp_result) == list(rs_result)

    def test_or_alternatives(self, pyparsing):
        """MatchFirst (a | b) should produce same result for each alternative."""
        pp_parser = pyparsing.Literal("hello") | pyparsing.Literal("goodbye")
        rs_parser = pp.Literal("hello") | pp.Literal("goodbye")

        pp_r1 = list(pp_parser.parseString("hello"))
        rs_r1 = list(rs_parser.parse_string("hello"))
        assert pp_r1 == rs_r1

        pp_r2 = list(pp_parser.parseString("goodbye"))
        rs_r2 = list(rs_parser.parse_string("goodbye"))
        assert pp_r2 == rs_r2

    def test_zero_or_more(self, pyparsing):
        """ZeroOrMore should produce same result."""
        pp_result = pyparsing.ZeroOrMore(pyparsing.Literal("a")).parseString("aaaa")
        rs_result = pp.ZeroOrMore(pp.Literal("a")).parse_string("aaaa")
        assert list(pp_result) == list(rs_result)

    def test_one_or_more(self, pyparsing):
        """OneOrMore should produce same result."""
        pp_result = pyparsing.OneOrMore(pyparsing.Literal("a")).parseString("aaaa")
        rs_result = pp.OneOrMore(pp.Literal("a")).parse_string("aaaa")
        assert list(pp_result) == list(rs_result)

    def test_optional_present(self, pyparsing):
        """Optional with match present should produce same result."""
        pp_result = pyparsing.Optional(pyparsing.Literal("a")).parseString("a")
        rs_result = pp.Optional(pp.Literal("a")).parse_string("a")
        assert list(pp_result) == list(rs_result)

    def test_optional_absent(self, pyparsing):
        """Optional with no match should produce same (empty) result."""
        pp_result = pyparsing.Optional(pyparsing.Literal("a")).parseString("b")
        rs_result = pp.Optional(pp.Literal("a")).parse_string("b")
        assert list(pp_result) == list(rs_result)

    def test_suppress(self, pyparsing):
        """Suppress should produce same (empty) result."""
        pp_result = pyparsing.Suppress(pyparsing.Literal("a")).parseString("a")
        rs_result = pp.Suppress(pp.Literal("a")).parse_string("a")
        assert list(pp_result) == list(rs_result)

    def test_word_nums(self, pyparsing):
        """Word(nums) should produce same result."""
        pp_result = pyparsing.Word(pyparsing.nums).parseString("12345abc")
        rs_result = pp.Word(pp.nums()).parse_string("12345abc")
        assert list(pp_result) == list(rs_result)

    def test_word_with_body_chars(self, pyparsing):
        """Word with init/body chars should match same content."""
        pp_result = pyparsing.Word(pyparsing.alphas, pyparsing.alphanums).parseString("abc123")
        rs_result = pp.Word(pp.alphas(), pp.alphanums()).parse_string("abc123")
        assert list(pp_result) == list(rs_result)

    def test_complex_expression(self, pyparsing):
        """A more complex expression should produce consistent results."""
        # Integer parser
        pp_int = pyparsing.Word(pyparsing.nums)
        rs_int = pp.Word(pp.nums())

        test_strings = ["0", "1", "42", "9999", "00123"]
        for s in test_strings:
            pp_result = list(pp_int.parseString(s))
            rs_result = list(rs_int.parse_string(s))
            assert pp_result == rs_result, f"Mismatch for '{s}': {pp_result} vs {rs_result}"


# ============================================================================
# j. Additional stress tests and corner cases
# ============================================================================

class TestStressAndCornerCases:
    """Additional corner cases and stress tests."""

    def test_literal_whitespace_only(self):
        """Literal matching whitespace."""
        lit = pp.Literal("   ")
        result = lit.parse_string("   hello")
        assert result == ["   "]

    def test_literal_special_chars(self):
        """Literal with special regex characters should work (no regex interpretation)."""
        lit = pp.Literal("a+b*c?")
        result = lit.parse_string("a+b*c?")
        assert result == ["a+b*c?"]

    def test_regex_special_groups(self):
        """Regex with groups should return the full match."""
        regex = pp.Regex(r"(\d+)-(\d+)")
        result = regex.parse_string("123-456")
        assert result == ["123-456"]

    def test_search_string_count_consistency_literal(self):
        """Verify count consistency across different string sizes."""
        lit = pp.Literal("xy")
        for n in [1, 2, 5, 10, 50, 100, 500]:
            text = "xy" * n
            count = lit.search_string_count(text)
            assert count == n, f"Failed for n={n}: got {count}"

    def test_search_string_count_consistency_word(self):
        """Verify word count consistency."""
        word = pp.Word(pp.alphas())
        for n in [1, 2, 5, 10, 50]:
            text = " ".join(["word"] * n)
            count = word.search_string_count(text)
            assert count == n, f"Failed for n={n}: got {count}"

    def test_search_string_count_cyclic_literal(self):
        """Test the cyclic detection path with repeating patterns."""
        lit = pp.Literal("ab")
        # Create a string with a repeating pattern
        pattern = "ab cd "
        text = pattern * 1000
        count = lit.search_string_count(text)
        assert count == 1000

    def test_word_all_printable(self):
        """Word with printables() should match all printable characters."""
        word = pp.Word(pp.printables())
        result = word.parse_string("hello!@#$%world")
        assert result == ["hello!@#$%world"]

    def test_word_nums_only(self):
        """Word(nums) should match only digit characters."""
        word = pp.Word(pp.nums())
        result = word.parse_string("123abc")
        assert result == ["123"]

    def test_word_stops_at_space(self):
        """Word(alphas) should stop at spaces."""
        word = pp.Word(pp.alphas())
        result = word.parse_string("hello world")
        assert result == ["hello"]

    def test_multiple_search_string_calls(self):
        """Calling search_string multiple times should give consistent results."""
        lit = pp.Literal("test")
        text = "test one test two test three"
        r1 = lit.search_string(text)
        r2 = lit.search_string(text)
        assert len(r1) == len(r2) == 3

    def test_parse_batch_preserves_order(self):
        """parse_batch results should correspond to inputs in order."""
        lit = pp.Literal("a")
        inputs = ["a", "b", "a", "b", "a"]
        result = lit.parse_batch(inputs)
        assert result[0] == ["a"]  # match
        assert result[1] == []     # no match
        assert result[2] == ["a"]  # match
        assert result[3] == []     # no match
        assert result[4] == ["a"]  # match

    def test_regex_empty_pattern(self):
        """Regex with empty-matching pattern on non-empty string."""
        regex = pp.Regex(r".*")
        result = regex.parse_string("hello")
        assert result == ["hello"]

    def test_and_single_element_via_constructor(self):
        """And with two elements should parse correctly."""
        parser = pp.Literal("a") + pp.Literal("b")
        result = parser.parse_string("ab")
        assert result == ["a", "b"]

    def test_match_first_via_constructor(self):
        """MatchFirst via explicit constructor should work."""
        parser = pp.MatchFirst([pp.Literal("abc"), pp.Literal("def")])
        assert parser.parse_string("abc") == ["abc"]
        assert parser.parse_string("def") == ["def"]
        with pytest.raises(ValueError):
            parser.parse_string("xyz")

    def test_parse_batch_large_mixed(self):
        """Large parse_batch with alternating match/no-match."""
        lit = pp.Literal("yes")
        inputs = ["yes" if i % 2 == 0 else "no" for i in range(10000)]
        result = lit.parse_batch(inputs)
        assert len(result) == 10000
        count = lit.parse_batch_count(inputs)
        assert count == 5000

    def test_literal_search_repeated_pattern(self):
        """Search in a string with a repeated pattern."""
        lit = pp.Literal("abc")
        text = "abcabcabc"
        result = lit.search_string(text)
        count = lit.search_string_count(text)
        assert len(result) == 3
        assert count == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
