"""Tests for moirais.fn.gibbs -- Gibbs sampler for bivariate normal."""

import numpy as np
from moirais.fn.gibbs import gibbs_bivariate_normal


def test_returns_dict():
    result = gibbs_bivariate_normal(n_iter=100)
    assert isinstance(result, dict)
    assert "samples" in result


def test_correct_shape():
    result = gibbs_bivariate_normal(n_iter=500)
    assert result["samples"].shape == (500, 2)


def test_burn_in():
    result = gibbs_bivariate_normal(n_iter=500, burn_in=100)
    assert result["samples"].shape == (400, 2)


def test_recovers_mean():
    result = gibbs_bivariate_normal(mu=(3.0, -2.0), n_iter=10000, burn_in=1000)
    means = result["sample_mean"]
    assert abs(means[0] - 3.0) < 0.5
    assert abs(means[1] - (-2.0)) < 0.5


def test_recovers_correlation():
    rho = 0.7
    result = gibbs_bivariate_normal(rho=rho, n_iter=10000, burn_in=2000)
    assert abs(result["sample_corr"] - rho) < 0.2


def test_invalid_rho():
    try:
        gibbs_bivariate_normal(rho=1.0)
        assert False
    except ValueError:
        pass


def test_invalid_sigma():
    try:
        gibbs_bivariate_normal(sigma=(0, 1))
        assert False
    except ValueError:
        pass
