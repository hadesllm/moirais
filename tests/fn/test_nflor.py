"""Test noise_floor (nflor)."""
import numpy as np
from moirais.fn.nflor import noise_floor, nflor
from moirais.fn._containers import DescriptiveResult


class TestNflor:
    def test_basic(self):
        x = np.random.default_rng(42).standard_normal(256)
        result = noise_floor(x)
        assert isinstance(result, DescriptiveResult)
        assert result.name == "noise_floor"

    def test_percentile(self):
        x = np.random.default_rng(42).standard_normal(256)
        result = noise_floor(x, method="percentile")
        assert result.extra["method"] == "percentile"

    def test_alias(self):
        assert nflor is noise_floor
