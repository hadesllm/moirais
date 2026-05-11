"""Test rr_variability (rrvar)."""
import numpy as np
from morie.fn.rrvar import rr_variability, rrvar
from morie.fn._containers import DescriptiveResult


class TestRrVariability:
    def test_basic(self):
        rr = np.array([0.8, 0.85, 0.78, 0.82, 0.9, 0.77, 0.83])
        result = rr_variability(rr)
        assert isinstance(result, DescriptiveResult)
        assert result.name == "rr_variability"

    def test_sdnn_positive(self):
        rr = np.array([0.8, 0.85, 0.78, 0.82, 0.9])
        result = rr_variability(rr)
        assert result.extra["sdnn"] > 0

    def test_rmssd_positive(self):
        rr = np.array([0.8, 0.85, 0.78, 0.82, 0.9])
        result = rr_variability(rr)
        assert result.extra["rmssd"] > 0

    def test_short_rr(self):
        result = rr_variability(np.array([0.8]))
        assert result.value == 0.0

    def test_alias(self):
        assert rrvar is rr_variability
