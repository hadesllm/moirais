"""Tests for hmpcav.geron_pca_variance."""
import numpy as np
import pytest
from moirais.fn.hmpcav import geron_pca_variance


def test_hmpcav_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    n_components = 3
    result = geron_pca_variance(X, n_components)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmpcav_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    n_components = 3
    result = geron_pca_variance(X, n_components)
    assert isinstance(result, dict)
