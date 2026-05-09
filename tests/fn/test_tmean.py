"""Tests for trimmed_mean."""
import numpy as np, pytest
from moirais.fn.tmean import trimmed_mean


class TestTrimmedMean:
    def test_basic(self):
        x = [1, 2, 3, 4, 5, 100]
        r = trimmed_mean(x, proportion=0.2)
        assert r.measure == "trimmed_mean"
        assert r.estimate < 20

    def test_no_trim(self):
        x = [1, 2, 3, 4, 5]
        r = trimmed_mean(x, proportion=0.0)
        assert r.estimate == pytest.approx(3.0)

    def test_bad_proportion(self):
        with pytest.raises(ValueError):
            trimmed_mean([1, 2, 3], proportion=0.6)

    def test_empty(self):
        with pytest.raises(ValueError):
            trimmed_mean([])
