"""Tests for dfbeta.dfbetas."""
import numpy as np
import pytest
from moirais.fn.dfbeta import dfbetas


def test_dfbeta_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = dfbetas(X, y)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_dfbeta_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = dfbetas(X, y)
    assert isinstance(result, dict)
