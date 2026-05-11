"""Tests for morie.fn.groot -- logistic growth model."""

import numpy as np
from morie.fn.groot import logistic_growth, groot
from morie.fn._containers import DescriptiveResult


class TestGroot:
    def test_alias(self):
        assert groot is logistic_growth

    def test_logistic_fit(self):
        t = np.linspace(0, 10, 50)
        y = 100 / (1 + np.exp(-1.5 * (t - 5)))
        r = logistic_growth(t, y)
        assert isinstance(r, DescriptiveResult)
        assert abs(r.value["K"] - 100) < 10
        assert r.value["r_squared"] > 0.99

    def test_returns_fitted(self):
        t = np.linspace(0, 5, 20)
        y = 50 / (1 + np.exp(-2 * (t - 2.5)))
        r = logistic_growth(t, y)
        assert len(r.value["fitted"]) == 20
