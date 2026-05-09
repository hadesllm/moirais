"""Tests for moirais.fn.tpscrs — crime severity."""

import pytest
from moirais.fn.tpscrs import tps_crime_severity
from moirais.fn._containers import ESRes


class TestCrimeSeverity:
    def test_basic(self):
        offenses = ["Murder", "Theft", "Theft"]
        weights = {"Murder": 100, "Theft": 10}
        r = tps_crime_severity(offenses, weights)
        assert isinstance(r, ESRes)
        assert r.estimate == pytest.approx(120 / 3)

    def test_default_weight(self):
        r = tps_crime_severity(["Unknown"], {})
        assert r.estimate == pytest.approx(1.0)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            tps_crime_severity([], {"A": 1})
