"""Tests for trimmed_mean."""
import numpy as np, pytest
from morie.fn.trimm import trimmed_mean

class TestTrimm:
    def test_symmetric(self):
        x = np.arange(1, 11, dtype=float)
        r = trimmed_mean(x, trim=0.1)
        assert r.estimate == pytest.approx(5.5, abs=0.5)

    def test_zero_trim(self):
        x = np.array([2, 4, 6], dtype=float)
        r = trimmed_mean(x, trim=0.0)
        assert r.estimate == pytest.approx(4.0)
