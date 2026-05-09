"""Tests for moirais.fn.bsurv -- Bayesian survival model."""

import numpy as np
from moirais.fn.bsurv import bayesian_survival


def test_returns_dict():
    result = bayesian_survival([1, 2, 3], [1, 1, 0])
    assert isinstance(result, dict)
    assert "posterior_mean" in result
    assert "hazard_samples" in result


def test_conjugate_update():
    times = [1.0, 2.0, 3.0]
    events = [1, 1, 1]
    result = bayesian_survival(times, events, prior_a=1, prior_b=1)
    np.testing.assert_allclose(result["posterior_a"], 4.0)
    np.testing.assert_allclose(result["posterior_b"], 7.0)


def test_censoring_effect():
    times = [1.0, 2.0, 3.0]
    result_all = bayesian_survival(times, [1, 1, 1], prior_a=1, prior_b=1)
    result_cens = bayesian_survival(times, [1, 0, 0], prior_a=1, prior_b=1)
    assert result_all["posterior_mean"] > result_cens["posterior_mean"]


def test_ci_contains_mean():
    result = bayesian_survival([1, 2, 3, 4, 5], [1, 1, 1, 1, 1])
    assert result["ci_lower"] < result["posterior_mean"] < result["ci_upper"]


def test_median_survival_positive():
    result = bayesian_survival([1, 2, 3], [1, 1, 1])
    assert result["median_survival"] > 0


def test_empty_times():
    try:
        bayesian_survival([], [])
        assert False
    except ValueError:
        pass


def test_mismatched_lengths():
    try:
        bayesian_survival([1, 2], [1])
        assert False
    except ValueError:
        pass
