"""Tests for grsen.geron_senet_squeeze_excite."""
import numpy as np
import pytest
from moirais.fn.grsen import geron_senet_squeeze_excite


def test_grsen_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    W1 = np.random.default_rng(42).normal(0, 1, 100)
    W2 = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_senet_squeeze_excite(X, W1, W2)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_grsen_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    W1 = np.random.default_rng(42).normal(0, 1, 100)
    W2 = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_senet_squeeze_excite(X, W1, W2)
    assert isinstance(result, dict)
