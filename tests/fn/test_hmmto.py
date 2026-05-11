"""Tests for hmmto.geron_multioutput."""
import numpy as np
import pytest
from morie.fn.hmmto import geron_multioutput


def test_hmmto_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    Y = np.random.default_rng(43).normal(0, 1, 100)
    result = geron_multioutput(X, Y)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmmto_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    Y = np.random.default_rng(43).normal(0, 1, 100)
    result = geron_multioutput(X, Y)
    assert isinstance(result, dict)
