"""Test calibration_plot (calpl)."""
import numpy as np
from moirais.fn.calpl import calibration_plot, calpl
from moirais.fn._containers import DescriptiveResult


class TestCalpl:
    def test_basic(self):
        rng = np.random.default_rng(42)
        y_true = rng.integers(0, 2, 100)
        y_prob = rng.uniform(0, 1, 100)
        result = calibration_plot(y_true, y_prob, n_bins=5)
        assert isinstance(result, DescriptiveResult)
        assert result.name == "calibration_plot"
        assert result.extra["ece"] >= 0

    def test_perfect_calibration(self):
        y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        y_prob = np.array([0.0, 0.1, 0.1, 0.1, 0.1, 0.9, 0.9, 0.9, 0.9, 1.0])
        result = calibration_plot(y_true, y_prob, n_bins=5)
        assert result.extra["ece"] < 0.3

    def test_alias(self):
        assert calpl is calibration_plot
