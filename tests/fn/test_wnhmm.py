"""Test hamming_window (wnhmm)."""
import numpy as np
from morie.fn.wnhmm import hamming_window, wnhmm
from morie.fn._containers import DescriptiveResult


class TestWnhmm:
    def test_basic(self):
        result = hamming_window(16)
        assert isinstance(result, DescriptiveResult)
        assert result.name == "hamming_window"

    def test_nonzero_endpoints(self):
        result = hamming_window(32)
        w = result.extra["window"]
        assert w[0] > 0.07

    def test_alias(self):
        assert wnhmm is hamming_window
