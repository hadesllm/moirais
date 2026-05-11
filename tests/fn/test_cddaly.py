"""Tests for morie.fn.cddaly -- DALY."""

import pytest
from morie.fn.cddaly import daly_calc


class TestDALY:
    def test_basic(self):
        res = daly_calc(yll=300, yld=100)
        assert res.estimate == pytest.approx(400.0)

    def test_pct(self):
        res = daly_calc(yll=300, yld=100)
        assert res.extra["pct_yll"] == pytest.approx(75.0)

    def test_negative_raises(self):
        with pytest.raises(ValueError):
            daly_calc(-1, 10)
