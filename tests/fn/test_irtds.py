"""Tests for moirais.fn.irtds — IRT discrimination extraction."""

import numpy as np
import pandas as pd
from moirais.fn.irtds import irt_discrimination


class TestIrtDiscrimination:
    def test_returns_dataframe(self):
        params = {"i1": {"a": 1.2, "b": -0.5}, "i2": {"a": 0.8, "b": 1.0}}
        result = irt_discrimination(params)
        assert isinstance(result, pd.DataFrame)
        assert "a" in result.columns
        assert len(result) == 2

    def test_correct_values(self):
        params = {"i1": {"a": 1.5}, "i2": {"a": 0.3}}
        result = irt_discrimination(params)
        assert result["a"].tolist() == [1.5, 0.3]

    def test_rasch_default_one(self):
        params = {"i1": {"b": 0.5}, "i2": {"b": -0.3}}
        result = irt_discrimination(params)
        assert (result["a"] == 1.0).all()
