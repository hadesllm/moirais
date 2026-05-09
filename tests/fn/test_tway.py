"""Tests for moirais.fn.tway -- Two-way fixed effects."""

import numpy as np
import pandas as pd
import pytest
from moirais.fn.tway import twfe, tway
from moirais.fn._containers import RegressionResult


class TestTWFE:
    def test_alias(self):
        assert tway is twfe

    def test_known_effect(self):
        """TWFE with constant treatment effect."""
        rng = np.random.default_rng(42)
        rows = []
        n_units = 20
        n_times = 10
        for i in range(n_units):
            unit_fe = rng.normal(0, 2)
            for t in range(n_times):
                time_fe = 0.5 * t
                treated = 1 if i < 10 and t >= 5 else 0
                y = unit_fe + time_fe + 3.0 * treated + rng.normal(0, 0.5)
                rows.append({"unit": i, "time": t, "outcome": y, "treatment": treated})
        df = pd.DataFrame(rows)
        result = twfe(df)
        assert isinstance(result, RegressionResult)
        assert abs(result.coefficients["treatment"] - 3.0) < 1.5

    def test_r_squared(self):
        rng = np.random.default_rng(42)
        rows = []
        for i in range(10):
            for t in range(5):
                y = 10 + i + t + (1 if i < 5 and t >= 3 else 0) + rng.normal(0, 0.1)
                rows.append({"unit": i, "time": t, "outcome": y,
                             "treatment": 1 if i < 5 and t >= 3 else 0})
        df = pd.DataFrame(rows)
        result = twfe(df)
        assert 0 <= result.r_squared <= 1

    def test_no_variation_raises(self):
        rows = [{"unit": 0, "time": t, "outcome": 1.0, "treatment": 0}
                for t in range(5)]
        df = pd.DataFrame(rows)
        with pytest.raises(ValueError, match="variation"):
            twfe(df)
