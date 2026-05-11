"""Tests for winsorized_mean."""
import numpy as np, pytest
from morie.fn.wmean import winsorized_mean


class TestWinsorizedMean:
    def test_outlier_resistance(self):
        x = [1, 2, 3, 4, 5, 100]
        r = winsorized_mean(x, proportion=0.2)
        assert r.measure == "winsorized_mean"
        assert r.estimate < 20

    def test_no_winsorize(self):
        x = [1, 2, 3, 4, 5]
        r = winsorized_mean(x, proportion=0.0)
        assert r.estimate == pytest.approx(3.0)

    def test_bad_proportion(self):
        with pytest.raises(ValueError):
            winsorized_mean([1, 2, 3], proportion=0.6)
