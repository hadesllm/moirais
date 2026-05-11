"""Tests for morie.fn.tpscmp — complaint rate."""

import pytest
from morie.fn.tpscmp import tps_complaint_rate
from morie.fn._containers import CrimeResult


class TestComplaintRate:
    def test_basic(self):
        r = tps_complaint_rate(50, 5000)
        assert isinstance(r, CrimeResult)
        assert r.rate == pytest.approx(10.0)

    def test_per_100(self):
        r = tps_complaint_rate(50, 5000, per=100)
        assert r.rate == pytest.approx(1.0)

    def test_invalid(self):
        with pytest.raises(ValueError):
            tps_complaint_rate(10, 0)
