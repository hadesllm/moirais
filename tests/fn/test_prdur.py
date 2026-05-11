"""Test pr_duration (prdur)."""
import numpy as np
from morie.fn.prdur import pr_duration, prdur
from morie.fn._containers import DescriptiveResult


class TestPrDuration:
    def test_basic(self):
        p_on = np.array([50, 450, 850])
        qrs_on = np.array([100, 500, 900])
        result = pr_duration(p_on, qrs_on, fs=250.0)
        assert isinstance(result, DescriptiveResult)
        assert result.name == "pr_duration"

    def test_correct_interval(self):
        p_on = np.array([0, 100])
        qrs_on = np.array([40, 140])
        result = pr_duration(p_on, qrs_on, fs=100.0)
        assert np.allclose(result.value, 0.4)

    def test_empty(self):
        result = pr_duration(np.array([]), np.array([]), fs=1.0)
        assert result.value == 0.0

    def test_alias(self):
        assert prdur is pr_duration
