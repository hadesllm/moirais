"""Tests for morie.fn.tpspr — patrol efficiency."""

import pytest
from morie.fn.tpspr import tps_patrol_efficiency
from morie.fn._containers import ESRes


class TestPatrolEfficiency:
    def test_basic(self):
        r = tps_patrol_efficiency(100, 5000)
        assert isinstance(r, ESRes)
        assert r.estimate == pytest.approx(0.02)

    def test_invalid(self):
        with pytest.raises(ValueError):
            tps_patrol_efficiency(10, 0)
