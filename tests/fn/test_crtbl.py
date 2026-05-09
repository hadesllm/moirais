"""Tests for moirais.fn.crtbl — bail grant rate."""

import pytest
from moirais.fn.crtbl import court_bail_rate
from moirais.fn._containers import CrimeResult


class TestBailRate:
    def test_basic(self):
        r = court_bail_rate(70, 100)
        assert r.rate == pytest.approx(0.7)
        assert r.extra["denial_rate"] == pytest.approx(0.3)

    def test_invalid(self):
        with pytest.raises(ValueError):
            court_bail_rate(5, 0)
