"""Tests for moirais.fn.xsrsk -- Excess risk."""

import pytest
from moirais.fn.xsrsk import excess_risk


class TestExcessRisk:
    def test_known(self):
        res = excess_risk(observed=100, expected=50.0)
        assert res.extra["excess_count"] == pytest.approx(50.0)

    def test_with_population(self):
        res = excess_risk(observed=100, expected=50.0, population=100000)
        assert res.extra["excess_rate_per"] > 0

    def test_smr(self):
        res = excess_risk(observed=100, expected=50.0)
        assert res.extra["SMR"] == pytest.approx(2.0)

    def test_invalid(self):
        with pytest.raises(ValueError):
            excess_risk(observed=-1, expected=50.0)
