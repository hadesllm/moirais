"""Tests for moirais.fn.crtbk — court backlog."""

import pytest
from moirais.fn.crtbk import court_backlog
from moirais.fn._containers import DescriptiveResult


class TestCourtBacklog:
    def test_basic(self):
        r = court_backlog(1000, 500, 600)
        assert isinstance(r, DescriptiveResult)
        assert r.extra["backlog_ratio"] == pytest.approx(2.0)

    def test_clearance(self):
        r = court_backlog(100, 200, 200)
        assert r.extra["clearance_rate"] == pytest.approx(1.0)
