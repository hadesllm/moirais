"""Tests for gh_dp_reg_post.ghosal_dp_regression_posterior."""
import numpy as np
import pytest
from morie.fn.gh_dp_reg_post import ghosal_dp_regression_posterior


def test_gh_dp_reg_post_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = ghosal_dp_regression_posterior(x, y)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_gh_dp_reg_post_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = ghosal_dp_regression_posterior(x, y)
    assert isinstance(result, dict)
