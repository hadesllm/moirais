"""Tests for morie.fn.mtowd — weather factor."""

import pytest
from morie.fn.mtowd import mto_weather_factor
from morie.fn._containers import DescriptiveResult


class TestWeatherFactor:
    def test_basic(self):
        r = mto_weather_factor([10, 5, 20, 8], ["Clear", "Rain", "Snow", "Fog"])
        assert isinstance(r, DescriptiveResult)
        assert r.extra["by_weather"]["Snow"] == 20

    def test_mismatch_raises(self):
        with pytest.raises(ValueError):
            mto_weather_factor([1, 2], ["A"])
