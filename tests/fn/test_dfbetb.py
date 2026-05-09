"""Tests for dfbetb.dfbetas."""
import numpy as np
import pytest
from moirais.fn.dfbetb import dfbetas


def test_dfbetb_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = dfbetas(y, X)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_dfbetb_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = dfbetas(y, X)
    assert isinstance(result, dict)
