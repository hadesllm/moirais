"""Tests for moirais.fn.loo -- LOO-CV via PSIS."""

import numpy as np
import pytest
from moirais.fn.loo import compute_loo


class TestLOO:
    def test_returns_expected_keys(self):
        rng = np.random.default_rng(42)
        ll = rng.normal(-1, 0.5, (100, 20))
        result = compute_loo(ll)
        for key in ("loo", "elpd_loo", "p_loo", "se", "k_hat"):
            assert key in result

    def test_loo_is_finite(self):
        rng = np.random.default_rng(42)
        ll = rng.normal(-2, 1, (50, 15))
        result = compute_loo(ll)
        assert np.isfinite(result["loo"])
        assert np.isfinite(result["se"])

    def test_k_hat_length_matches_obs(self):
        rng = np.random.default_rng(42)
        n_obs = 25
        ll = rng.normal(-1, 0.5, (80, n_obs))
        result = compute_loo(ll)
        assert len(result["k_hat"]) == n_obs

    def test_too_few_samples_raises(self):
        with pytest.raises(ValueError, match="10"):
            compute_loo(np.ones((5, 10)))
