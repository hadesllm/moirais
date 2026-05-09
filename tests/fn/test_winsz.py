"""Tests for winsorize."""
import numpy as np, pytest
from moirais.fn.winsz import winsorize

class TestWinsorize:
    def test_basic(self):
        x = np.arange(1, 21, dtype=float)
        r = winsorize(x, trim=0.1)
        assert r.measure == "winsorized_mean"
        assert 5 < r.estimate < 15

    def test_no_trim(self):
        x = np.array([1, 2, 3, 4, 5], dtype=float)
        r = winsorize(x, trim=0.0)
        assert r.estimate == pytest.approx(3.0)
