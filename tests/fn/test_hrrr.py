"""Test heart_rate_from_rr (hrrr)."""
import numpy as np
import pytest

from morie.fn.hrrr import heart_rate_from_rr, hrrr
from morie.fn._containers import DescriptiveResult


class TestHeartRateFromRR:
    def test_basic(self):
        rr = np.array([0.8, 0.85, 0.75, 0.9])
        result = heart_rate_from_rr(rr)
        assert isinstance(result, DescriptiveResult)
        assert result.name == "heart_rate_from_rr"

    def test_known_values(self):
        rr = np.array([1.0, 0.5])
        result = heart_rate_from_rr(rr)
        hr = result.extra["heart_rates"]
        np.testing.assert_allclose(hr, [60.0, 120.0])

    def test_mean_hr(self):
        rr = np.array([1.0, 0.5])
        result = heart_rate_from_rr(rr)
        assert result.extra["mean_hr"] == pytest.approx(90.0)

    def test_alias(self):
        assert hrrr is heart_rate_from_rr
