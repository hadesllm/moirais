"""Tests for partial_kendall."""
import numpy as np, pytest
from morie.fn.pknds import partial_kendall


class TestPartialKendall:
    def test_basic(self):
        rng = np.random.default_rng(42)
        z = rng.normal(0, 1, 50)
        x = z + rng.normal(0, 0.5, 50)
        y = z + rng.normal(0, 0.5, 50)
        r = partial_kendall(x, y, z)
        assert r.measure == "partial_kendall"
        assert abs(r.estimate) < abs(r.extra["tau_xy"])

    def test_unrelated(self):
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, 30)
        y = rng.normal(0, 1, 30)
        z = rng.normal(0, 1, 30)
        r = partial_kendall(x, y, z)
        assert abs(r.estimate) < 0.5

    def test_too_few(self):
        with pytest.raises(ValueError):
            partial_kendall([1, 2], [3, 4], [5, 6])
