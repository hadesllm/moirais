"""Tests for morie.fn.stdd -- Staggered DiD."""

import numpy as np
import pandas as pd
import pytest
from morie.fn.stdd import staggered_did, stdd
from morie.fn._containers import ESRes


def _make_panel(n_units=20, n_times=10, treat_effect=3.0, seed=42):
    """Create panel with staggered adoption."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_units):
        if i < 5:
            g = 5
        elif i < 10:
            g = 7
        else:
            g = np.inf
        for t in range(1, n_times + 1):
            treated = 1 if t >= g else 0
            y = 10 + rng.normal(0, 1) + treat_effect * treated
            rows.append({"unit": i, "time": t, "outcome": y, "treat_time": g})
    return pd.DataFrame(rows)


class TestStaggeredDiD:
    def test_alias(self):
        assert stdd is staggered_did

    def test_known_effect(self):
        df = _make_panel(treat_effect=3.0)
        result = staggered_did(df)
        assert isinstance(result, ESRes)
        assert abs(result.estimate - 3.0) < 2.0

    def test_multiple_cohorts(self):
        df = _make_panel()
        result = staggered_did(df)
        assert result.extra["n_cohorts"] == 2

    def test_no_valid_cohorts_raises(self):
        df = pd.DataFrame({
            "unit": [1, 1], "time": [1, 2],
            "outcome": [1.0, 2.0], "treat_time": [np.inf, np.inf]
        })
        with pytest.raises(ValueError, match="No valid"):
            staggered_did(df)
