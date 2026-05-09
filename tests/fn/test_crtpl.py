"""Tests for moirais.fn.crtpl — plea rate."""

import pytest
from moirais.fn.crtpl import court_plea_rate
from moirais.fn._containers import CrimeResult


class TestPleaRate:
    def test_basic(self):
        pleas = ["Guilty"] * 70 + ["Not guilty"] * 30
        r = court_plea_rate(pleas)
        assert isinstance(r, CrimeResult)
        assert r.rate == pytest.approx(0.7)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            court_plea_rate([])
