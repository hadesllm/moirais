"""Tests for moirais.fn.zoomv -- Velocity profile."""

import numpy as np
import pandas as pd
from moirais.fn.zoomv import velocity_profile, zoomv
from moirais.fn._containers import DescriptiveResult


class TestZoomv:
    def test_alias(self):
        assert zoomv is velocity_profile

    def test_linear(self):
        df = pd.DataFrame({"x": np.arange(10, dtype=float)})
        result = velocity_profile(df, dt=1.0)
        assert isinstance(result, DescriptiveResult)
        assert all(abs(v - 1.0) < 0.01 for v in result.value["velocity"][1:-1])

    def test_keys(self):
        df = pd.DataFrame({"x": np.sin(np.linspace(0, 2 * np.pi, 50))})
        result = velocity_profile(df)
        assert "velocity" in result.value
        assert "acceleration" in result.value
        assert "jerk" in result.value
