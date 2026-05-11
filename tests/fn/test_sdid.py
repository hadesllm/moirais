"""Tests for morie.fn.sdid -- Synthetic DiD."""

import numpy as np
import pandas as pd
import pytest
from morie.fn.sdid import synthetic_did, sdid
from morie.fn._containers import ESRes


def _make_sdid_panel(n_ctrl=8, n_pre=6, n_post=3, effect=4.0, seed=42):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_ctrl + 1):
        is_treated = (i == 0)
        for t in range(1, n_pre + n_post + 1):
            post = (t > n_pre)
            base = 10 + 0.5 * t + rng.normal(0, 1)
            y = base + effect * is_treated * post
            d = 1 if is_treated and post else 0
            rows.append({
                "unit": f"u{i}", "time": t, "outcome": y, "treatment": d
            })
    return pd.DataFrame(rows)


class TestSyntheticDiD:
    def test_alias(self):
        assert sdid is synthetic_did

    def test_known_effect(self):
        df = _make_sdid_panel(effect=4.0)
        result = synthetic_did(df)
        assert isinstance(result, ESRes)
        assert abs(result.estimate - 4.0) < 4.0

    def test_output_structure(self):
        df = _make_sdid_panel()
        result = synthetic_did(df)
        assert result.extra["n_treated_units"] == 1
        assert result.extra["n_control_units"] >= 1
        assert result.extra["n_pre_periods"] >= 2

    def test_no_control_raises(self):
        rows = [{"unit": "u0", "time": t, "outcome": float(t),
                 "treatment": 1 if t > 3 else 0}
                for t in range(1, 7)]
        df = pd.DataFrame(rows)
        with pytest.raises(ValueError, match="control"):
            synthetic_did(df)
