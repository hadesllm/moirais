"""Tests for moirais.fn.grndr -- Grenander estimator."""

import numpy as np
import pytest
from moirais.fn.grndr import grenander_estimator


class TestGrenander:
    def test_basic_output(self):
        rng = np.random.default_rng(42)
        x = rng.exponential(1.0, 200)
        r = grenander_estimator(x)
        assert len(r["knots"]) > 0
        assert len(r["heights"]) == len(r["knots"])

    def test_heights_non_increasing(self):
        rng = np.random.default_rng(42)
        x = rng.exponential(1.0, 500)
        r = grenander_estimator(x)
        h = np.array(r["heights"])
        assert np.all(np.diff(h) <= 1e-10)

    def test_lcm_is_concave(self):
        rng = np.random.default_rng(42)
        x = rng.exponential(1.0, 100)
        r = grenander_estimator(x)
        lcm_x = np.array(r["lcm_x"])
        lcm_y = np.array(r["lcm_y"])
        for i in range(1, len(lcm_x) - 1):
            dx1 = lcm_x[i] - lcm_x[i - 1]
            dy1 = lcm_y[i] - lcm_y[i - 1]
            dx2 = lcm_x[i + 1] - lcm_x[i]
            dy2 = lcm_y[i + 1] - lcm_y[i]
            if dx1 > 0 and dx2 > 0:
                assert dy1 / dx1 >= dy2 / dx2 - 1e-10

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            grenander_estimator(np.array([]))

    def test_negative_data_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            grenander_estimator(np.array([-1.0, 2.0, 3.0]))
