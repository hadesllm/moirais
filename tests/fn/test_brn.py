"""Tests for morie.fn.brn — Brunner-Munzel test."""
import numpy as np
import pytest
from morie.fn.brn import brunner_munzel


class TestBrunnerMunzel:
    def test_basic(self):
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, 50)
        y = rng.normal(0.5, 1.5, 50)
        res = brunner_munzel(x, y)
        assert isinstance(res.extra["statistic"], float)
        assert 0 <= res.extra["p_value"] <= 1

    def test_p_hat_range(self):
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, 60)
        y = rng.normal(0, 1, 60)
        res = brunner_munzel(x, y)
        assert 0 <= res.extra["p_hat"] <= 1

    def test_identical_samples(self):
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, 40)
        y = rng.normal(0, 1, 40)
        res = brunner_munzel(x, y)
        assert abs(res.extra["p_hat"] - 0.5) < 0.15
