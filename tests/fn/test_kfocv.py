"""Tests for morie.fn.kfocv -- K-fold cross-validation."""

import numpy as np
import pandas as pd
from morie.fn.kfocv import kfold_cv, kfocv
from morie.fn._containers import DescriptiveResult


class TestKfocv:
    def test_alias(self):
        assert kfocv is kfold_cv

    def test_linear_data_low_mse(self):
        """Linear relationship: CV MSE should be close to noise variance."""
        rng = np.random.default_rng(42)
        n = 500
        x = rng.normal(0, 1, n)
        y = 3 * x + 2 + rng.normal(0, 1, n)  # noise var = 1
        df = pd.DataFrame({"y": y, "x": x})
        result = kfold_cv(df, k=5)
        assert isinstance(result, DescriptiveResult)
        assert result.value < 2.0  # MSE should be near 1

    def test_k_folds_reported(self):
        rng = np.random.default_rng(42)
        n = 200
        df = pd.DataFrame({"y": rng.normal(0, 1, n), "x": rng.normal(0, 1, n)})
        result = kfold_cv(df, k=10)
        assert result.extra["k"] == 10
        assert len(result.extra["fold_mses"]) == 10

    def test_multiple_predictors(self):
        rng = np.random.default_rng(42)
        n = 300
        x1 = rng.normal(0, 1, n)
        x2 = rng.normal(0, 1, n)
        y = 2 * x1 + x2 + rng.normal(0, 1, n)
        df = pd.DataFrame({"y": y, "x1": x1, "x2": x2})
        result = kfold_cv(df, x=["x1", "x2"])
        assert result.value < 3.0
