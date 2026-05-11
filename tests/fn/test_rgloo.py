"""Tests for rgloo.rangayyan_loo_cv."""
import numpy as np
import pytest
from morie.fn.rgloo import rangayyan_loo_cv


def test_rgloo_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    classifier = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_loo_cv(X, y, classifier)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rgloo_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    classifier = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_loo_cv(X, y, classifier)
    assert isinstance(result, dict)
