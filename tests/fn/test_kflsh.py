"""Tests for moirais.fn.kflsh -- Acceleration profile."""

import numpy as np
import pandas as pd
from moirais.fn.kflsh import acceleration_profile, kflsh
from moirais.fn._containers import DescriptiveResult


class TestKflsh:
    def test_alias(self):
        assert kflsh is acceleration_profile

    def test_constant_velocity(self):
        df = pd.DataFrame({"x": np.ones(20) * 5.0})
        result = acceleration_profile(df, dt=1.0)
        assert isinstance(result, DescriptiveResult)
        assert result.value < 0.01

    def test_accelerating(self):
        df = pd.DataFrame({"x": np.arange(10, dtype=float) ** 2})
        result = acceleration_profile(df, dt=1.0)
        assert result.value > 0
