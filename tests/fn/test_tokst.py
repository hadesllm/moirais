"""Test token_stats."""
import numpy as np
from moirais.fn.tokst import token_stats, tokst
from moirais.fn._containers import DescriptiveResult


class TestTokenStats:
    def test_basic(self):
        result = token_stats([1, 2, 3, 1, 2, 1], vocab_size=100)
        assert isinstance(result, DescriptiveResult)
        assert result.name == "token_stats"

    def test_entropy(self):
        result = token_stats([0] * 100, vocab_size=100)
        assert result.value < 0.01

    def test_unique(self):
        result = token_stats([1, 2, 3, 4, 5], vocab_size=10)
        assert result.extra["unique_tokens"] == 5

    def test_alias(self):
        assert tokst is token_stats
