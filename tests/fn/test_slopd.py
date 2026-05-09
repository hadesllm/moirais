"""Test slope_detect (slopd)."""
import numpy as np
from moirais.fn.slopd import slope_detect, slopd
from moirais.fn._containers import DescriptiveResult


class TestSlopeDetect:
    def test_basic(self):
        t = np.linspace(0, 2 * np.pi, 200)
        x = np.sin(t)
        result = slope_detect(x, fs=100.0)
        assert isinstance(result, DescriptiveResult)
        assert result.name == "slope_detect"

    def test_detects_events(self):
        x = np.zeros(200)
        x[50:60] = np.linspace(0, 10, 10)
        x[60:70] = np.linspace(10, 0, 10)
        result = slope_detect(x, fs=1.0, threshold=0.5)
        assert result.value >= 1

    def test_derivative_length(self):
        x = np.random.default_rng(42).standard_normal(100)
        result = slope_detect(x, fs=1.0)
        assert len(result.extra["derivative"]) == len(x) - 1

    def test_alias(self):
        assert slopd is slope_detect
