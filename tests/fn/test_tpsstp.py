"""Tests for morie.fn.tpsstp — stop and search."""

import pytest
from morie.fn.tpsstp import tps_stop_and_search
from morie.fn._containers import DescriptiveResult


class TestStopAndSearch:
    def test_equal_rates(self):
        stops = {"A": 100, "B": 100}
        pop = {"A": 10000, "B": 10000}
        r = tps_stop_and_search(stops, pop)
        assert isinstance(r, DescriptiveResult)
        assert r.extra["disparity_ratios"]["A"] == pytest.approx(1.0)

    def test_disparity(self):
        stops = {"A": 100, "B": 300}
        pop = {"A": 10000, "B": 10000}
        r = tps_stop_and_search(stops, pop)
        assert r.extra["disparity_ratios"]["B"] > r.extra["disparity_ratios"]["A"]

    def test_empty_pop_raises(self):
        with pytest.raises(ValueError):
            tps_stop_and_search({"A": 10}, {"A": 0})
