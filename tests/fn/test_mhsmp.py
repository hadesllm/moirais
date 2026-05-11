"""Tests for morie.fn.mhsmp -- Metropolis-Hastings sampler."""

import numpy as np
from morie.fn.mhsmp import metropolis_hastings


def test_returns_dict():
    result = metropolis_hastings(lambda x: -0.5 * float(x @ x), [0.0, 0.0], n_iter=100)
    assert isinstance(result, dict)
    assert "samples" in result


def test_correct_shape():
    result = metropolis_hastings(lambda x: -0.5 * float(x @ x), [0.0], n_iter=500)
    assert result["samples"].shape == (500, 1)


def test_burn_in_and_thin():
    result = metropolis_hastings(
        lambda x: -0.5 * float(x @ x), [0.0], n_iter=1000, burn_in=200, thin=2
    )
    assert result["samples"].shape == (400, 1)


def test_acceptance_rate_reasonable():
    result = metropolis_hastings(lambda x: -0.5 * float(x @ x), [0.0, 0.0], n_iter=5000)
    assert 0.05 < result["acceptance_rate"] < 0.95


def test_samples_near_target_mean():
    result = metropolis_hastings(
        lambda x: -0.5 * float(x @ x), [0.0], n_iter=10000, burn_in=2000
    )
    assert abs(np.mean(result["samples"])) < 0.5


def test_invalid_burn_in():
    try:
        metropolis_hastings(lambda x: 0.0, [0.0], n_iter=10, burn_in=10)
        assert False
    except ValueError:
        pass


def test_reproducibility():
    f = lambda x: -0.5 * float(x @ x)
    r1 = metropolis_hastings(f, [1.0], n_iter=100, seed=99)
    r2 = metropolis_hastings(f, [1.0], n_iter=100, seed=99)
    np.testing.assert_array_equal(r1["samples"], r2["samples"])
