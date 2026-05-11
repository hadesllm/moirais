"""Tests for morie.fn.tpsgis — geo analysis."""

import pytest
import numpy as np
from morie.fn.tpsgis import tps_geo_analysis
from morie.fn._containers import DescriptiveResult


class TestGeoAnalysis:
    def test_basic(self):
        rng = np.random.default_rng(42)
        r = tps_geo_analysis(rng.normal(43.65, 0.05, 100), rng.normal(-79.38, 0.05, 100))
        assert isinstance(r, DescriptiveResult)
        assert r.extra["centroid_lat"] == pytest.approx(43.65, abs=0.1)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            tps_geo_analysis([], [])
