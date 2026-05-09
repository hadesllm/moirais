"""Tests for moirais.fn.tpstrn — crime trend."""

import pytest
import numpy as np
from moirais.fn.tpstrn import tps_crime_trend
from moirais.fn._containers import DescriptiveResult


class TestCrimeTrend:
    def test_increasing(self):
        r = tps_crime_trend([10, 20, 30, 40, 50])
        assert isinstance(r, DescriptiveResult)
        assert r.extra["slope"] > 0
        assert r.extra["trend"] == "increasing"

    def test_stable(self):
        rng = np.random.default_rng(42)
        r = tps_crime_trend(rng.normal(100, 1, 20))
        assert r.extra["trend"] == "stable"

    def test_too_short(self):
        with pytest.raises(ValueError):
            tps_crime_trend([1, 2])
