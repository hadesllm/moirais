"""Tests for morie.fn.inc -- Incidence rate."""

import pytest
from morie.fn.inc import incidence_rate


class TestIncidenceRate:
    def test_known_rate(self):
        result = incidence_rate(50, 1000.0)
        assert result["rate"] == pytest.approx(0.05)

    def test_ci_brackets_rate(self):
        result = incidence_rate(100, 5000.0)
        assert result["ci_lower"] < result["rate"] < result["ci_upper"]

    def test_zero_events(self):
        result = incidence_rate(0, 1000.0)
        assert result["rate"] == 0.0
        assert result["ci_lower"] == 0.0
        assert result["ci_upper"] > 0.0

    def test_negative_events_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            incidence_rate(-1, 100.0)

    def test_zero_person_time_raises(self):
        with pytest.raises(ValueError, match="positive"):
            incidence_rate(5, 0.0)
