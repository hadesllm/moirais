"""Tests for moirais.fn.csdid -- Callaway-Sant'Anna DiD."""

import numpy as np
import pandas as pd
import pytest
from moirais.fn.csdid import cs_did, csdid


def _make_cs_panel(n_units=30, n_times=8, effect=2.0, seed=42):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_units):
        if i < 8:
            g = 4
        elif i < 16:
            g = 6
        else:
            g = np.inf
        for t in range(1, n_times + 1):
            treated = 1 if t >= g and np.isfinite(g) else 0
            y = 5 + rng.normal(0, 1) + effect * treated
            rows.append({"unit": i, "time": t, "outcome": y, "treat_time": g})
    return pd.DataFrame(rows)


class TestCSDiD:
    def test_alias(self):
        assert csdid is cs_did

    def test_known_effect(self):
        df = _make_cs_panel(effect=2.0)
        result = cs_did(df)
        assert abs(result["att_agg"] - 2.0) < 2.0
        assert len(result["att_gt"]) > 0

    def test_multiple_cohorts(self):
        df = _make_cs_panel()
        result = cs_did(df)
        assert result["n_cohorts"] == 2
        assert result["n_cells"] >= 2

    def test_no_never_treated_raises(self):
        rows = [{"unit": i, "time": t, "outcome": 1.0, "treat_time": 3}
                for i in range(5) for t in range(1, 6)]
        df = pd.DataFrame(rows)
        with pytest.raises(ValueError, match="never-treated"):
            cs_did(df)
