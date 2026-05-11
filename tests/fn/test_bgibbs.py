"""Tests for morie.fn.bgibbs — Gibbs sampler for normal."""

import numpy as np
import pytest

from morie.fn.bgibbs import gibbs_normal


class TestGibbsNormal:
    def test_posterior_near_data_mean(self):
        rng = np.random.default_rng(42)
        data = rng.normal(5, 2, 100)
        res = gibbs_normal(data, n_iter=3000, seed=42)
        assert abs(res.value - 5.0) < 1.0

    def test_sigma2_positive(self):
        rng = np.random.default_rng(42)
        data = rng.normal(0, 3, 50)
        res = gibbs_normal(data, n_iter=2000, seed=42)
        assert res.extra["posterior_sigma2_mean"] > 0

    def test_output_lengths(self):
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        res = gibbs_normal(data, n_iter=1000, seed=42)
        assert len(res.extra["mu_samples"]) == 1000
        assert len(res.extra["sigma2_samples"]) == 1000
