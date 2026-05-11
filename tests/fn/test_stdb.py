"""Tests for morie.fn.stdb -- standardized regression coefficients."""

import numpy as np
import pandas as pd
import pytest
from morie.fn.stdb import standardized_coefficients


class TestStandardizedCoefficients:
    def test_single_predictor(self, rng):
        """Strong linear relationship should give large beta."""
        n = 200
        x = rng.standard_normal(n)
        y = 3.0 * x + rng.standard_normal(n) * 0.3
        X = pd.DataFrame({"x": x})
        result = standardized_coefficients(X, y)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1
        assert abs(result.iloc[0]["beta"]) > 0.8

    def test_multiple_predictors(self, rng):
        """Two predictors should return two rows."""
        n = 200
        x1 = rng.standard_normal(n)
        x2 = rng.standard_normal(n)
        y = 2.0 * x1 + 0.5 * x2 + rng.standard_normal(n) * 0.3
        X = pd.DataFrame({"x1": x1, "x2": x2})
        result = standardized_coefficients(X, y)
        assert len(result) == 2
        assert "variable" in result.columns
        assert "beta" in result.columns
        assert "p_value" in result.columns
