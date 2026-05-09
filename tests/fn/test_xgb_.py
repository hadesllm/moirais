"""Tests for moirais.fn.xgb_ — simplified XGBoost."""
import numpy as np
from moirais.fn.xgb_ import xgboost_simple


class TestXGBoost:
    def test_basic(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((80, 3))
        y = X @ [1, 2, 0] + rng.standard_normal(80) * 0.5
        res = xgboost_simple(X, y, n_trees=20, max_depth=2)
        assert res.value > 0

    def test_predictions_shape(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((30, 2))
        y = rng.standard_normal(30)
        res = xgboost_simple(X, y, n_trees=5)
        assert len(res.extra["predictions"]) == 30
