"""Tests for moirais.fn.irtdl — IRT difficulty extraction."""

import numpy as np
import pandas as pd
from moirais.fn.irtdl import irt_difficulty


class TestIrtDifficulty:
    def test_returns_dataframe(self):
        params = {"i1": {"a": 1.2, "b": -0.5}, "i2": {"a": 0.8, "b": 1.0}}
        result = irt_difficulty(params)
        assert isinstance(result, pd.DataFrame)
        assert "b" in result.columns
        assert len(result) == 2

    def test_correct_values(self):
        params = {"i1": {"b": -1.0}, "i2": {"b": 0.5}, "i3": {"b": 2.0}}
        result = irt_difficulty(params)
        assert result["b"].tolist() == [-1.0, 0.5, 2.0]

    def test_grm_thresholds(self):
        params = {"i1": {"b": [-1.0, 0.0, 1.0]}}
        result = irt_difficulty(params)
        assert len(result) == 3
        assert "i1_t1" in result["item"].values

    def test_rasch_default_a(self):
        params = {"i1": {"b": 0.5}}
        result = irt_difficulty(params)
        assert result.loc[0, "b"] == 0.5
