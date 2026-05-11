"""Tests for morie.fn.hedgw — Edgeworth expansion for KDE."""

import numpy as np
import pytest

from morie.fn.hedgw import hedgw


class TestHedgw:
    def test_f_hat_positive(self):
        rng = np.random.default_rng(42)
        data = rng.normal(0, 1, 200)
        res = hedgw(data, x0=0.0)
        assert res["f_hat"] > 0

    def test_ci_contains_f_hat(self):
        rng = np.random.default_rng(42)
        data = rng.normal(0, 1, 200)
        res = hedgw(data, x0=0.0)
        assert res["ci_lower"] <= res["f_hat"] <= res["ci_upper"]

    def test_variance_positive(self):
        rng = np.random.default_rng(42)
        data = rng.normal(0, 1, 100)
        res = hedgw(data, x0=0.0)
        assert res["variance"] > 0

    def test_f_hat_near_true(self):
        rng = np.random.default_rng(42)
        data = rng.normal(0, 1, 1000)
        res = hedgw(data, x0=0.0)
        true_val = 1.0 / np.sqrt(2 * np.pi)
        assert res["f_hat"] == pytest.approx(true_val, abs=0.05)

    def test_raises_small(self):
        with pytest.raises(ValueError):
            hedgw(np.array([1.0, 2.0]), x0=0.0)
