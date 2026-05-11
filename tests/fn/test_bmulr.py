"""Tests for morie.fn.bmulr -- Bayesian multinomial."""

import numpy as np
from morie.fn.bmulr import bayesian_multinomial


def test_returns_dict():
    result = bayesian_multinomial([10, 20, 30])
    assert isinstance(result, dict)
    assert "posterior_mean" in result


def test_posterior_mean_sums_to_one():
    result = bayesian_multinomial([10, 20, 30])
    assert abs(sum(result["posterior_mean"]) - 1.0) < 1e-10


def test_largest_category():
    result = bayesian_multinomial([1, 1, 100])
    assert result["posterior_mean"][2] > result["posterior_mean"][0]
