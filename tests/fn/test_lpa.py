"""Tests for latent_profile."""
import numpy as np, pytest
from morie.fn.lpa import latent_profile

class TestLPA:
    def test_two_profiles(self):
        rng = np.random.default_rng(0)
        X = np.vstack([rng.normal(0, 1, (30, 3)), rng.normal(5, 1, (30, 3))])
        r = latent_profile(X, n_profiles=2, seed=0)
        assert r.name == "lpa"
        assert sum(r.extra["profile_sizes"]) == 60

    def test_bic(self):
        rng = np.random.default_rng(1)
        X = rng.normal(0, 1, (40, 2))
        r = latent_profile(X, n_profiles=2, seed=1)
        assert np.isfinite(r.value)
