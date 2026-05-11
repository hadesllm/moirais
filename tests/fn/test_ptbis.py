"""Tests for point_biserial."""
import numpy as np, pytest
from morie.fn.ptbis import point_biserial


class TestPointBiserial:
    def test_correlated(self):
        x = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
        y = [1, 2, 1, 2, 1, 8, 9, 8, 9, 8]
        r = point_biserial(x, y)
        assert r.measure == "point_biserial"
        assert r.estimate > 0.5

    def test_no_correlation(self):
        rng = np.random.default_rng(42)
        x = rng.choice([0, 1], 100)
        y = rng.normal(0, 1, 100)
        r = point_biserial(x, y)
        assert abs(r.estimate) < 0.3

    def test_not_binary(self):
        with pytest.raises(ValueError):
            point_biserial([1, 2, 3], [4, 5, 6])
