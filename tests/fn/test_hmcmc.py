"""Tests for morie.fn.hmcmc -- Hamiltonian Monte Carlo."""

import numpy as np
from morie.fn.hmcmc import hamiltonian_mc


def _log_std_normal(x):
    return -0.5 * float(x @ x)


def _grad_log_std_normal(x):
    return -x


def test_returns_dict():
    result = hamiltonian_mc(_log_std_normal, _grad_log_std_normal, [0.0], n_iter=100)
    assert isinstance(result, dict)
    assert "samples" in result


def test_samples_shape():
    result = hamiltonian_mc(_log_std_normal, _grad_log_std_normal, [0.0, 0.0], n_iter=200)
    assert result["samples"].shape == (200, 2)


def test_acceptance_rate():
    result = hamiltonian_mc(_log_std_normal, _grad_log_std_normal, [0.0], n_iter=500)
    assert result["acceptance_rate"] > 0
