"""Tests for moirais.fn.rkhs -- RKHS regression."""

import numpy as np
import pytest
from moirais.fn.rkhs import rkhs


class TestRkhs:
    def test_perfect_prediction(self):
        rng = np.random.default_rng(42)
        n = 30
        K = np.eye(n)
        y = rng.standard_normal(n)
        res = rkhs(y, K, lambda_val=0.001)
        assert res.statistic > 0.9

    def test_returns_correct_name(self):
        K = np.eye(10)
        y = np.arange(10, dtype=float)
        res = rkhs(y, K)
        assert res.name == "RKHS"

    def test_fitted_length(self):
        K = np.eye(10)
        y = np.arange(10, dtype=float)
        res = rkhs(y, K)
        assert len(res.extra["fitted"]) == 10

    def test_negative_lambda_raises(self):
        with pytest.raises(ValueError):
            rkhs(np.ones(5), np.eye(5), lambda_val=-1)

    def test_dimension_mismatch(self):
        with pytest.raises(ValueError):
            rkhs(np.ones(5), np.eye(3))
