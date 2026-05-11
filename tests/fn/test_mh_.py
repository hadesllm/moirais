"""Tests for morie.fn.mh_ -- Metropolis-Hastings sampler."""

import numpy as np
import pytest
from morie.fn.mh_ import metropolis_hastings


class TestMetropolisHastings:
    def test_samples_converge_to_target_mean(self):
        """MH sampling from N(3, 1) should have mean near 3."""
        def log_post(x):
            return -0.5 * (x - 3.0) ** 2

        result = metropolis_hastings(log_post, start=0.0, n_iter=20000, proposal_sd=1.0, seed=42)
        # Burn-in: discard first 5000
        samples = result["samples"][5000:]
        assert abs(np.mean(samples) - 3.0) < 0.2

    def test_acceptance_rate_reasonable(self):
        """Acceptance rate should be between 0.1 and 0.9 for well-tuned proposal."""
        def log_post(x):
            return -0.5 * x ** 2

        result = metropolis_hastings(log_post, n_iter=10000, proposal_sd=1.5, seed=42)
        assert 0.1 < result["acceptance_rate"] < 0.9

    def test_deterministic_with_seed(self):
        """Same seed -> same samples."""
        def log_post(x):
            return -0.5 * x ** 2

        r1 = metropolis_hastings(log_post, seed=123, n_iter=100)
        r2 = metropolis_hastings(log_post, seed=123, n_iter=100)
        np.testing.assert_array_equal(r1["samples"], r2["samples"])

    def test_invalid_n_iter_raises(self):
        with pytest.raises(ValueError, match="n_iter"):
            metropolis_hastings(lambda x: -x ** 2, n_iter=0)
