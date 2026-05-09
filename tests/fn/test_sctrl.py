"""Tests for moirais.fn.sctrl -- Synthetic control."""

import numpy as np
import pandas as pd
import pytest
from moirais.fn.sctrl import synthetic_control, sctrl
from moirais.fn._containers import ESRes


def _make_sc_panel(n_ctrl=5, n_pre=8, n_post=4, effect=5.0, seed=42):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_ctrl + 1):
        unit_name = f"u{i}"
        for t in range(1, n_pre + n_post + 1):
            base = 10 + 0.3 * t + rng.normal(0, 0.5)
            if i == 0 and t > n_pre:
                base += effect
            rows.append({"unit": unit_name, "time": t, "outcome": base})
    return pd.DataFrame(rows)


class TestSyntheticControl:
    def test_alias(self):
        assert sctrl is synthetic_control

    def test_known_effect(self):
        df = _make_sc_panel(effect=5.0)
        result = synthetic_control(df, treated_unit="u0", treat_time=9)
        assert isinstance(result, ESRes)
        assert abs(result.estimate - 5.0) < 4.0

    def test_weights_sum_to_one(self):
        df = _make_sc_panel()
        result = synthetic_control(df, treated_unit="u0", treat_time=9)
        w_sum = sum(result.extra["weights"].values())
        assert abs(w_sum - 1.0) < 1e-5

    def test_pre_rmse_small(self):
        """Good synthetic control should have small pre-treatment RMSE."""
        df = _make_sc_panel(n_ctrl=10)
        result = synthetic_control(df, treated_unit="u0", treat_time=9)
        assert result.extra["pre_rmse"] < 5.0

    def test_requires_treat_time(self):
        df = _make_sc_panel()
        with pytest.raises(ValueError, match="treat_time"):
            synthetic_control(df, treated_unit="u0")

    def test_gaps_correct_length(self):
        df = _make_sc_panel(n_post=3)
        result = synthetic_control(df, treated_unit="u0", treat_time=9)
        assert len(result.extra["gaps"]) == 3
