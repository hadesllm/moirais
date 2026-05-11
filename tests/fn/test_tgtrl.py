"""Tests for morie.fn.tgtrl — Target trial emulation."""

import pandas as pd
import pytest

from morie.fn.tgtrl import target_trial


class TestTargetTrial:
    def test_basic(self):
        df = pd.DataFrame({
            "treatment": [1, 1, 1, 0, 0, 0, 1, 0, 1, 0],
            "outcome": [1, 1, 0, 0, 0, 1, 1, 0, 0, 0],
        })
        res = target_trial(df)
        assert "risk_difference" in res.extra

    def test_with_eligibility(self):
        df = pd.DataFrame({
            "treatment": [1, 1, 0, 0, 1, 0, 1, 0],
            "outcome": [1, 0, 0, 0, 1, 1, 0, 0],
            "age": [25, 30, 20, 50, 35, 40, 28, 45],
        })
        res = target_trial(df, eligibility_fn=lambda r: r["age"] < 40)
        assert res.extra["n_eligible"] < len(df)

    def test_missing_col(self):
        df = pd.DataFrame({"x": [1, 2]})
        with pytest.raises(ValueError):
            target_trial(df)
