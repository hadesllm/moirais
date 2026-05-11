"""Tests for morie.fn.mhsrv -- service utilization."""

import pytest
from morie.fn.mhsrv import service_utilization


class TestServiceUtilization:
    def test_basic(self):
        res = service_utilization(n_accessing=30, n_need=100)
        assert res.estimate == pytest.approx(0.3)
        assert res.extra["treatment_gap"] == pytest.approx(0.7)

    def test_invalid(self):
        with pytest.raises(ValueError):
            service_utilization(10, 0)
