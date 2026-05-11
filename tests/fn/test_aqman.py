"""Tests for morie.fn.aqman -- IDW interpolation."""

import numpy as np
import pandas as pd
from morie.fn.aqman import idw_interpolate, aqman
from morie.fn._containers import DescriptiveResult


class TestAqman:
    def test_alias(self):
        assert aqman is idw_interpolate

    def test_basic_interp(self):
        df = pd.DataFrame({
            "x": [0, 1, 0, 1, 0.5],
            "y": [0, 0, 1, 1, 0.5],
            "z": [0, 1, 1, 2, 1],
        })
        result = idw_interpolate(df, grid_size=5)
        assert isinstance(result, DescriptiveResult)
        assert result.extra["n_points"] == 5

    def test_grid_shape(self):
        rng = np.random.default_rng(42)
        df = pd.DataFrame({"x": rng.uniform(0, 10, 20), "y": rng.uniform(0, 10, 20), "z": rng.normal(0, 1, 20)})
        result = idw_interpolate(df, grid_size=10)
        assert len(result.value) == 10
        assert len(result.value[0]) == 10
