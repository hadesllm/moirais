"""Tests for morie.fn.evstd -- Event study design."""

import numpy as np
import pandas as pd
from morie.fn.evstd import event_study, evstd


class TestEventStudy:
    def test_alias(self):
        assert evstd is event_study

    def test_pre_trends_zero(self):
        """Pre-treatment coefficients should be near zero."""
        rng = np.random.default_rng(42)
        rows = []
        for i in range(30):
            g = 6 if i < 15 else np.inf
            for t in range(1, 11):
                treated = 1 if np.isfinite(g) and t >= g else 0
                y = 10 + rng.normal(0, 1) + 3.0 * treated
                rows.append({"unit": i, "time": t, "outcome": y, "treat_time": g})
        df = pd.DataFrame(rows)
        result = event_study(df, pre_window=4, post_window=4)
        for k in [-4, -3, -2]:
            if k in result["coefficients"]:
                assert abs(result["coefficients"][k]) < 3.0

    def test_ref_period_is_zero(self):
        rng = np.random.default_rng(42)
        rows = []
        for i in range(20):
            g = 5 if i < 10 else np.inf
            for t in range(1, 9):
                treated = 1 if np.isfinite(g) and t >= g else 0
                y = rng.normal(0, 1) + 2.0 * treated
                rows.append({"unit": i, "time": t, "outcome": y, "treat_time": g})
        df = pd.DataFrame(rows)
        result = event_study(df, ref_period=-1)
        assert result["coefficients"][-1] == 0.0
        assert result["ref_period"] == -1

    def test_post_coefficients_positive(self):
        """Post-treatment coefficients should capture the effect."""
        rng = np.random.default_rng(42)
        rows = []
        for i in range(30):
            g = 5 if i < 15 else np.inf
            for t in range(1, 10):
                treated = 1 if np.isfinite(g) and t >= g else 0
                y = 5 + rng.normal(0, 0.5) + 4.0 * treated
                rows.append({"unit": i, "time": t, "outcome": y, "treat_time": g})
        df = pd.DataFrame(rows)
        result = event_study(df, pre_window=3, post_window=4)
        post_coefs = [result["coefficients"][k] for k in range(0, 5) if k in result["coefficients"]]
        if post_coefs:
            assert np.mean(post_coefs) > 0
