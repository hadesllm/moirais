"""Test st_segment (stsgm)."""
import numpy as np
from moirais.fn.stsgm import st_segment, stsgm
from moirais.fn._containers import DescriptiveResult


class TestStSegment:
    def test_basic(self):
        signal = np.random.default_rng(42).standard_normal(1000)
        qrs_off = np.array([130, 530])
        t_on = np.array([180, 580])
        result = st_segment(signal, qrs_off, t_on, fs=250.0)
        assert isinstance(result, DescriptiveResult)
        assert result.name == "st_segment"

    def test_levels_computed(self):
        signal = np.ones(500)
        qrs_off = np.array([100, 300])
        t_on = np.array([150, 350])
        result = st_segment(signal, qrs_off, t_on, fs=250.0)
        assert np.allclose(result.extra["st_levels"], 1.0)

    def test_empty(self):
        result = st_segment(np.ones(100), np.array([]), np.array([]), fs=1.0)
        assert result.value == 0.0

    def test_alias(self):
        assert stsgm is st_segment
