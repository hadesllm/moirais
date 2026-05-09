"""Test pp_interval (ppint)."""
import numpy as np
from moirais.fn.ppint import pp_interval, ppint
from moirais.fn._containers import DescriptiveResult


class TestPpInterval:
    def test_basic(self):
        peaks = np.array([100, 200, 310, 400])
        result = pp_interval(peaks, fs=250.0)
        assert isinstance(result, DescriptiveResult)
        assert result.name == "pp_interval"

    def test_correct_intervals(self):
        peaks = np.array([0, 100, 200, 300])
        result = pp_interval(peaks, fs=100.0)
        assert np.allclose(result.extra["pp_intervals"], 1.0)

    def test_single_peak(self):
        result = pp_interval(np.array([50]), fs=1.0)
        assert result.value == 0.0

    def test_alias(self):
        assert ppint is pp_interval
