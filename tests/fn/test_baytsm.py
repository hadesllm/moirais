"""Tests for baytsm.bayes_time_series."""
import numpy as np
import pytest
from moirais.fn.baytsm import bayes_time_series


def test_baytsm_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    model = np.random.default_rng(42).normal(0, 1, 100)
    priors = np.random.default_rng(42).normal(0, 1, 100)
    result = bayes_time_series(y, model, priors)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_baytsm_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    model = np.random.default_rng(42).normal(0, 1, 100)
    priors = np.random.default_rng(42).normal(0, 1, 100)
    result = bayes_time_series(y, model, priors)
    assert isinstance(result, dict)
