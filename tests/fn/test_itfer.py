"""Tests for morie.fn.itfer — interference effects."""
import numpy as np
import pytest
from morie.fn.itfer import interference_effects


class TestInterferenceEffects:
    def test_clustered_data(self):
        rng = np.random.default_rng(42)
        n = 200
        clusters = np.repeat(np.arange(20), 10)
        treatment = rng.choice([0, 1], size=n)
        y = treatment * 0.5 + rng.standard_normal(n) * 0.3
        res = interference_effects(y, treatment, clusters)
        assert isinstance(res.extra["direct_effect"], float)
        assert res.extra["n_clusters"] == 20

    def test_n_correct(self):
        rng = np.random.default_rng(42)
        n = 100
        clusters = np.repeat(np.arange(10), 10)
        treatment = rng.choice([0, 1], size=n)
        y = rng.standard_normal(n)
        res = interference_effects(y, treatment, clusters)
        assert res.extra["n"] == n
