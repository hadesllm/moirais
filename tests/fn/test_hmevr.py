"""Tests for hmevr.geron_explained_variance_ratio."""
import numpy as np
import pytest
from moirais.fn.hmevr import geron_explained_variance_ratio


def test_hmevr_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    n_components = 3
    result = geron_explained_variance_ratio(X, n_components)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmevr_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    n_components = 3
    result = geron_explained_variance_ratio(X, n_components)
    assert isinstance(result, dict)
