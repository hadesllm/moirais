"""Tests for moirais.fn.aqidx -- air quality index."""

import pytest
from moirais.fn.aqidx import air_quality_index


class TestAQI:
    def test_good(self):
        res = air_quality_index({"pm25": 8.0})
        assert res.extra["category"] == "good"

    def test_moderate(self):
        res = air_quality_index({"pm25": 20.0})
        assert res.extra["category"] == "moderate"

    def test_empty(self):
        res = air_quality_index({})
        assert res.estimate == 0.0
