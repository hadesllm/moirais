"""Test peak_detect_threshold (pdthr)."""
import numpy as np
from morie.fn.pdthr import peak_detect_threshold, pdthr
from morie.fn._containers import DescriptiveResult


class TestPeakDetectThreshold:
    def test_basic(self):
        t = np.linspace(0, 4 * np.pi, 200)
        x = np.sin(t)
        result = peak_detect_threshold(x, threshold=0.5, min_dist=10)
        assert isinstance(result, DescriptiveResult)
        assert result.name == "peak_detect_threshold"

    def test_finds_peaks(self):
        t = np.linspace(0, 4 * np.pi, 200)
        x = np.sin(t)
        result = peak_detect_threshold(x, threshold=0.5, min_dist=10)
        assert result.value >= 1

    def test_amplitudes(self):
        t = np.linspace(0, 4 * np.pi, 200)
        x = np.sin(t)
        result = peak_detect_threshold(x, threshold=0.5, min_dist=10)
        for a in result.extra["amplitudes"]:
            assert a >= 0.5

    def test_alias(self):
        assert pdthr is peak_detect_threshold
