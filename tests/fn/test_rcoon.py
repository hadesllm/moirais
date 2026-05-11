"""Tests for morie.fn.rcoon -- Hohmann transfer orbit."""

from morie.fn.rcoon import hohmann_transfer, rcoon
from morie.fn._containers import DescriptiveResult


class TestRcoon:
    def test_alias(self):
        assert rcoon is hohmann_transfer

    def test_leo_to_geo(self):
        r1 = 6.571e6
        r2 = 42.164e6
        r = hohmann_transfer(r1, r2)
        assert isinstance(r, DescriptiveResult)
        assert r.value["delta_v_total"] > 0
        assert r.value["transfer_time"] > 0

    def test_symmetric(self):
        r1 = hohmann_transfer(1e7, 2e7)
        r2 = hohmann_transfer(2e7, 1e7)
        assert abs(r1.value["delta_v_total"] - r2.value["delta_v_total"]) < 1
