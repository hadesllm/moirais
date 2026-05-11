"""Tests for morie.fn.bmcmc — Metropolis-Hastings sampler."""

import numpy as np
import pytest

from morie.fn.bmcmc import metropolis_hastings


class TestMetropolisHastings:
    def test_normal_target(self):
        def log_target(x):
            return -0.5 * x ** 2

        res = metropolis_hastings(log_target, 0.0, n_iter=10000, seed=42)
        assert abs(res.value) < 0.5

    def test_acceptance_rate(self):
        def log_target(x):
            return -0.5 * x ** 2

        res = metropolis_hastings(log_target, 0.0, n_iter=5000, proposal_sd=1.0, seed=42)
        assert 0.1 < res.extra["acceptance_rate"] < 0.9

    def test_samples_length(self):
        def log_target(x):
            return -0.5 * (x - 3) ** 2

        res = metropolis_hastings(log_target, 0.0, n_iter=2000, seed=42)
        assert len(res.extra["samples"]) == 2000
