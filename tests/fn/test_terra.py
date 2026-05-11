"""Tests for morie.fn.terra -- Terrain analysis."""

import numpy as np
from morie.fn.terra import terrain_analysis, terra
from morie.fn._containers import DescriptiveResult


class TestTerra:
    def test_alias(self):
        assert terra is terrain_analysis

    def test_flat(self):
        Z = np.ones((5, 5))
        result = terrain_analysis(Z)
        assert isinstance(result, DescriptiveResult)
        assert result.extra["mean_slope"] < 0.01

    def test_tilted(self):
        Z = np.tile(np.arange(10, dtype=float), (10, 1))
        result = terrain_analysis(Z, cell_size=1.0)
        assert result.extra["mean_slope"] > 0
