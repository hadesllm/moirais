"""Tests for morie.fn.hmc -- Hamiltonian Monte Carlo."""

import numpy as np
from morie.fn.hmc import hamiltonian_mc


def _log_normal(x):
    return -0.5 * float(x @ x)


def _grad_normal(x):
    return -x


def test_returns_dict():
    result = hamiltonian_mc(_log_normal, _grad_normal, [0.0, 0.0], n_iter=50)
    assert isinstance(result, dict)
    assert "samples" in result
    assert "acceptance_rate" in result


def test_correct_shape():
    result = hamiltonian_mc(_log_normal, _grad_normal, [0.0], n_iter=200)
    assert result["samples"].shape == (200, 1)


def test_high_acceptance():
    result = hamiltonian_mc(
        _log_normal, _grad_normal, [0.0, 0.0],
        n_iter=500, epsilon=0.05, n_leapfrog=20
    )
    assert result["acceptance_rate"] > 0.4


def test_samples_near_target_mean():
    result = hamiltonian_mc(
        _log_normal, _grad_normal, [0.0, 0.0],
        n_iter=2000, epsilon=0.05, n_leapfrog=20
    )
    means = np.mean(result["samples"][500:], axis=0)
    assert np.all(np.abs(means) < 0.5)


def test_invalid_epsilon():
    try:
        hamiltonian_mc(_log_normal, _grad_normal, [0.0], epsilon=0)
        assert False
    except ValueError:
        pass


def test_reproducibility():
    r1 = hamiltonian_mc(_log_normal, _grad_normal, [1.0], n_iter=50, seed=7)
    r2 = hamiltonian_mc(_log_normal, _grad_normal, [1.0], n_iter=50, seed=7)
    np.testing.assert_array_equal(r1["samples"], r2["samples"])
