"""Tests for morie.fn.brcc — Bayesian reliable change index."""

import numpy as np
import pytest
from morie.fn.brcc import bayesian_rci


class TestBayesianRci:

    def test_returns_dict(self, rng):
        pre = rng.normal(50, 10, 50)
        post = pre + rng.normal(5, 3, 50)
        result = bayesian_rci(pre, post, n_iter=200)
        assert "rci_classical" in result and "prob_reliable_change" in result

    def test_rci_array_length(self, rng):
        pre = rng.normal(50, 10, 30)
        post = pre + rng.normal(0, 2, 30)
        result = bayesian_rci(pre, post, n_iter=200)
        assert len(result["rci_classical"]) == 30
        assert len(result["prob_reliable_change"]) == 30

    def test_counts_sum_to_n(self, rng):
        pre = rng.normal(50, 10, 40)
        post = pre + rng.normal(5, 5, 40)
        result = bayesian_rci(pre, post, n_iter=200)
        total = result["n_improved"] + result["n_deteriorated"] + result["n_unchanged"]
        assert total == 40

    def test_sem_provided(self, rng):
        pre = rng.normal(50, 10, 50)
        post = pre + rng.standard_normal(50)
        result = bayesian_rci(pre, post, sem=3.0, n_iter=200)
        assert result["sem"] == 3.0

    def test_small_n(self):
        result = bayesian_rci(np.array([1.0, 2.0]), np.array([3.0, 4.0]), n_iter=100)
        assert result["n"] == 2
