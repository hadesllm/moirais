"""Tests for morie.fn.htidx — Heat index."""

import pytest

from morie.fn.htidx import heat_index


class TestHeatIndex:
    def test_low_temp(self):
        res = heat_index(70, 50)
        assert res.value < 80

    def test_high_temp_high_humidity(self):
        res = heat_index(100, 80)
        assert res.value > 100

    def test_celsius_conversion(self):
        res = heat_index(90, 50)
        hi_c = res.extra["heat_index_C"]
        assert isinstance(hi_c, float)
        assert hi_c < 50
