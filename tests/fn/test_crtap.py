"""Tests for morie.fn.crtap — appeal rate."""

import pytest
from morie.fn.crtap import court_appeal_rate
from morie.fn._containers import DescriptiveResult


class TestAppealRate:
    def test_basic(self):
        r = court_appeal_rate(50, 1000, 10)
        assert isinstance(r, DescriptiveResult)
        assert r.extra["appeal_rate"] == pytest.approx(0.05)
        assert r.extra["overturn_rate"] == pytest.approx(0.2)

    def test_no_appeals(self):
        r = court_appeal_rate(0, 100, 0)
        assert r.extra["overturn_rate"] == pytest.approx(0.0)
