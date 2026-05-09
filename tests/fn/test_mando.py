"""Tests for moirais.fn.mando -- Propensity score matching."""

import numpy as np
import pandas as pd
from moirais.fn.mando import ps_match, mando
from moirais.fn._containers import ESRes


class TestMando:
    def test_alias(self):
        assert mando is ps_match

    def test_matching_with_effect(self):
        """With confounding, PS matching should recover approximate ATT."""
        rng = np.random.default_rng(42)
        n = 600
        x = rng.normal(0, 1, n)
        # Treatment depends on x (confounding)
        p_treat = 1 / (1 + np.exp(-x))
        t = (rng.uniform(size=n) < p_treat).astype(float)
        # Outcome depends on x and treatment (true effect = 3)
        y = 2 * x + 3 * t + rng.normal(0, 1, n)
        df = pd.DataFrame({"outcome": y, "treatment": t, "x": x})
        result = ps_match(df, x="x", caliper=0.5)
        assert isinstance(result, ESRes)
        assert result.measure == "ATT (PS matching)"
        # Matching should bring estimate closer to 3 than naive diff
        assert abs(result.estimate - 3.0) < 2.0
        assert result.extra["n_matched"] > 10

    def test_n_matched_reported(self):
        rng = np.random.default_rng(42)
        n = 200
        x = rng.normal(0, 1, n)
        t = (rng.uniform(size=n) < 0.5).astype(float)
        y = x + t + rng.normal(0, 0.5, n)
        df = pd.DataFrame({"outcome": y, "treatment": t, "x": x})
        result = ps_match(df, x="x", caliper=0.5)
        assert result.n == result.extra["n_matched"] * 2
