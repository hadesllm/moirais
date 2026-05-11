"""Tests for morie.fn.crtpl — plea rate."""

import pytest
from morie.fn.crtpl import court_plea_rate
from morie.fn._containers import CrimeResult


class TestPleaRate:
    def test_basic(self):
        pleas = ["Guilty"] * 70 + ["Not guilty"] * 30
        r = court_plea_rate(pleas)
        assert isinstance(r, CrimeResult)
        assert r.rate == pytest.approx(0.7)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            court_plea_rate([])
