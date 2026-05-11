"""Tests for irtgu -- guessing parameter analysis."""
import numpy as np
from morie.fn.irtgu import irt_guessing
from morie.fn._containers import DescriptiveResult


class TestIrtGuessing:
    def test_basic(self):
        params = {f"item_{j}": {"a": 1.0, "b": 0.0, "c": 0.2} for j in range(5)}
        result = irt_guessing(params, n_options=4)
        assert isinstance(result, DescriptiveResult)
        assert result.extra["mean_c"] == 0.2

    def test_above_chance(self):
        params = {"i1": {"c": 0.3}, "i2": {"c": 0.1}}
        result = irt_guessing(params, n_options=4)
        assert result.extra["n_above_chance"] == 1
