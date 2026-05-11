"""Tests for morie.fn.ridlr -- Cipher frequency analysis."""

from morie.fn.ridlr import cipher_frequency, ridlr
from morie.fn._containers import DescriptiveResult


class TestRidlr:
    def test_alias(self):
        assert ridlr is cipher_frequency

    def test_english_text(self):
        text = "The quick brown fox jumps over the lazy dog"
        result = cipher_frequency(text)
        assert isinstance(result, DescriptiveResult)
        assert result.extra["n_letters"] == 35
        assert result.extra["n_unique"] > 20

    def test_repeated_char(self):
        text = "aaaaaaaaaa"
        result = cipher_frequency(text)
        assert result.value > 100
