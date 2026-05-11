"""Tests for gpcgs.gp_classification_svgp."""
import numpy as np
import pytest
from morie.fn.gpcgs import gp_classification_svgp


def test_gpcgs_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    X_test = np.random.default_rng(43).normal(0, 1, 30)
    M = np.random.default_rng(43).normal(0, 1, (10, 10))
    result = gp_classification_svgp(X, y, X_test, M)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_gpcgs_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    X_test = np.random.default_rng(43).normal(0, 1, 30)
    M = np.random.default_rng(43).normal(0, 1, (10, 10))
    result = gp_classification_svgp(X, y, X_test, M)
    assert isinstance(result, dict)
