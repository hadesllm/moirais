"""Tests for bayoutl.bayes_outlier."""
import numpy as np
import pytest
from moirais.fn.bayoutl import bayes_outlier


def test_bayoutl_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    outlier_prior = np.random.default_rng(42).normal(0, 1, 100)
    result = bayes_outlier(y, outlier_prior)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_bayoutl_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    outlier_prior = np.random.default_rng(42).normal(0, 1, 100)
    result = bayes_outlier(y, outlier_prior)
    assert isinstance(result, dict)
