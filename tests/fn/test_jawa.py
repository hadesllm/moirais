"""Tests for morie.fn.jawa -- Jackknife estimator."""

import numpy as np
from morie.fn.jawa import jackknife, jawa
from morie.fn._containers import DescriptiveResult


class TestJawa:
    def test_alias(self):
        assert jawa is jackknife

    def test_mean_se_reasonable(self):
        """Jackknife SE of the mean should approximate analytical SE."""
        rng = np.random.default_rng(42)
        x = rng.normal(10, 2, 100)
        result = jackknife(x)
        assert isinstance(result, DescriptiveResult)
        assert result.name == "Jackknife"
        analytical_se = x.std(ddof=1) / np.sqrt(len(x))
        assert abs(result.extra["se"] - analytical_se) < 0.05

    def test_bias_near_zero_for_mean(self):
        """For the mean, jackknife bias should be near zero."""
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, 50)
        result = jackknife(x)
        assert abs(result.extra["bias"]) < 0.01

    def test_handles_nan(self):
        x = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
        result = jackknife(x)
        assert result.extra["n"] == 4
