"""Tests for moirais.fn.force -- ATE difference-in-means."""

import numpy as np
import pandas as pd
from moirais.fn.force import ate_diff, force
from moirais.fn._containers import ESRes


class TestForce:
    def test_alias(self):
        assert force is ate_diff

    def test_known_ate(self):
        """Treatment adds exactly 5; ATE should be close to 5."""
        rng = np.random.default_rng(42)
        n = 500
        t = np.concatenate([np.ones(250), np.zeros(250)])
        y = rng.normal(0, 1, n) + 5 * t
        df = pd.DataFrame({"outcome": y, "treatment": t})
        result = ate_diff(df)
        assert isinstance(result, ESRes)
        assert result.measure == "ATE (diff-in-means)"
        assert abs(result.estimate - 5.0) < 0.5
        assert result.ci_lower < 5.0 < result.ci_upper

    def test_no_effect(self):
        """When treatment has no effect, ATE should be near zero."""
        rng = np.random.default_rng(42)
        n = 400
        t = np.concatenate([np.ones(200), np.zeros(200)])
        y = rng.normal(10, 2, n)
        df = pd.DataFrame({"outcome": y, "treatment": t})
        result = ate_diff(df)
        assert abs(result.estimate) < 1.0
        assert result.ci_lower < 0 < result.ci_upper

    def test_extra_fields(self):
        rng = np.random.default_rng(42)
        t = np.array([1, 1, 1, 0, 0, 0], dtype=float)
        y = rng.normal(0, 1, 6)
        df = pd.DataFrame({"outcome": y, "treatment": t})
        result = ate_diff(df)
        assert result.extra["n1"] == 3
        assert result.extra["n0"] == 3
        assert "df" in result.extra
