"""Tests for moirais.fn.dqerr — dequantization error metrics."""

import numpy as np
import pytest

from moirais.fn.dqerr import dequant_error


class TestDequantError:

    def test_perfect(self):
        x = np.random.default_rng(42).standard_normal(64)
        res = dequant_error(x, x)
        assert res.value == pytest.approx(0.0, abs=1e-15)
        assert res.extra["cosine_similarity"] == pytest.approx(1.0, abs=1e-10)
        assert res.extra["sqnr_db"] == float("inf")

    def test_nonzero_error(self):
        x = np.random.default_rng(0).standard_normal(64)
        x_hat = x + 0.1
        res = dequant_error(x, x_hat)
        assert res.value > 0
        assert res.extra["cosine_similarity"] < 1.0
        assert res.extra["sqnr_db"] > 0

    def test_mismatched_raises(self):
        with pytest.raises(ValueError):
            dequant_error(np.ones(10), np.ones(20))

    def test_rmse(self):
        x = np.ones(100)
        x_hat = x + 0.5
        res = dequant_error(x, x_hat)
        assert res.extra["rmse"] == pytest.approx(0.5, abs=1e-10)
