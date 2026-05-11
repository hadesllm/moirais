"""Tests for morie.fn.vctrp — repeat victimization."""

import pytest
from morie.fn.vctrp import victim_repeat
from morie.fn._containers import DescriptiveResult


class TestVictimRepeat:
    def test_from_ids(self):
        r = victim_repeat(["V1", "V1", "V2", "V3", "V3", "V3"])
        assert isinstance(r, DescriptiveResult)
        assert r.extra["n_repeat"] == 2

    def test_from_counts(self):
        r = victim_repeat([], incident_counts=[1, 2, 3, 1, 1])
        assert r.extra["pct_repeat"] == pytest.approx(0.4)
