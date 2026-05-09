"""Tests for moirais.fn.ihgap -- health gap."""

import pytest
from moirais.fn.ihgap import health_gap


class TestHealthGap:
    def test_basic(self):
        res = health_gap(rate_indigenous=0.15, rate_general=0.10)
        assert res.estimate == pytest.approx(0.05)
        assert res.extra["relative_gap_pct"] == pytest.approx(50.0)

    def test_no_gap(self):
        res = health_gap(0.10, 0.10)
        assert res.estimate == pytest.approx(0.0)
