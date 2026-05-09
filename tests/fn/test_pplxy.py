"""Test perplexity."""
import numpy as np
from moirais.fn.pplxy import perplexity, pplxy
from moirais.fn._containers import DescriptiveResult


class TestPerplexity:
    def test_basic(self):
        result = perplexity(1.0)
        assert isinstance(result, DescriptiveResult)
        assert result.name == "perplexity"

    def test_value(self):
        result = perplexity(1.0)
        assert abs(result.value - np.e) < 1e-6

    def test_zero_loss(self):
        result = perplexity(0.0)
        assert abs(result.value - 1.0) < 1e-10

    def test_alias(self):
        assert pplxy is perplexity
