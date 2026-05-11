"""Test zero_cross_detect (zcdet)."""
import numpy as np
from morie.fn.zcdet import zero_cross_detect, zcdet
from morie.fn._containers import DescriptiveResult


class TestZeroCrossDetect:
    def test_basic(self):
        t = np.linspace(0, 4 * np.pi, 200)
        x = np.sin(t)
        result = zero_cross_detect(x)
        assert isinstance(result, DescriptiveResult)
        assert result.name == "zero_cross_detect"

    def test_sine_crossings(self):
        t = np.linspace(0, 2 * np.pi, 1000)
        x = np.sin(t)
        result = zero_cross_detect(x)
        assert result.value >= 1

    def test_directions(self):
        x = np.array([-1, 1, -1, 1, -1], dtype=float)
        result = zero_cross_detect(x)
        assert result.value == 4
        assert result.extra["n_positive"] >= 1
        assert result.extra["n_negative"] >= 1

    def test_short_signal(self):
        result = zero_cross_detect(np.array([1.0]))
        assert result.value == 0.0

    def test_alias(self):
        assert zcdet is zero_cross_detect
