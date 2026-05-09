"""Tests for moirais.fn.dalyc -- DALY computation."""

import pytest
from moirais.fn.dalyc import daly_computation


class TestDALYC:
    def test_sum(self):
        res = daly_computation(yll=100.0, yld=50.0)
        assert res.estimate == pytest.approx(150.0)

    def test_pct(self):
        res = daly_computation(yll=100.0, yld=50.0)
        assert res.extra["yll_pct"] == pytest.approx(100 * 100 / 150)

    def test_negative(self):
        with pytest.raises(ValueError):
            daly_computation(yll=-10, yld=5)
