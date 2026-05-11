"""Tests for median_abs_dev."""
import numpy as np, pytest
from morie.fn.medab import median_abs_dev


class TestMedianAbsDev:
    def test_constant(self):
        r = median_abs_dev([5, 5, 5, 5])
        assert r.estimate == pytest.approx(0.0)

    def test_normal_scale(self):
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, 1000)
        r = median_abs_dev(x)
        assert r.estimate == pytest.approx(1.0, abs=0.2)

    def test_empty(self):
        with pytest.raises(ValueError):
            median_abs_dev([])
