"""Tests for moirais.fn.zatna -- Palindrome detection."""

from moirais.fn.zatna import palindrome_detect, zatna
from moirais.fn._containers import DescriptiveResult


class TestZatna:
    def test_alias(self):
        assert zatna is palindrome_detect

    def test_numeric_palindrome(self):
        seq = [1, 2, 3, 2, 1]
        result = palindrome_detect(seq, min_length=3)
        assert isinstance(result, DescriptiveResult)
        assert result.value >= 1

    def test_string_palindrome(self):
        seq = list("racecar")
        result = palindrome_detect(seq, min_length=3)
        assert result.value >= 1
        assert result.extra["longest"] >= 7
