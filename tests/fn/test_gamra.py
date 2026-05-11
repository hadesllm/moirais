"""Tests for morie.fn.gamra -- AFT survival model."""

import numpy as np
from morie.fn.gamra import aft_model, gamra
from morie.fn._containers import SurvivalResult


class TestGamra:
    def test_alias(self):
        assert gamra is aft_model

    def test_lognormal(self):
        rng = np.random.default_rng(42)
        n = 100
        t = rng.lognormal(1, 0.5, n)
        event = rng.binomial(1, 0.8, n)
        r = aft_model(t, event)
        assert isinstance(r, SurvivalResult)
        assert r.n_events + r.n_censored == n
        assert "aic" in r.extra

    def test_with_covariates(self):
        rng = np.random.default_rng(0)
        n = 80
        X = rng.normal(0, 1, (n, 2))
        t = np.exp(1 + 0.5 * X[:, 0] + rng.normal(0, 0.3, n))
        event = np.ones(n, dtype=int)
        r = aft_model(t, event, X=X)
        assert "coefficients" in r.extra
