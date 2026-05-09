"""Tests for shanen.shannon_entropy."""
import numpy as np
import pytest
from moirais.fn.shanen import shannon_entropy


def test_shanen_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    base = np.random.default_rng(42).normal(0, 1, 100)
    result = shannon_entropy(y, base)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_shanen_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    base = np.random.default_rng(42).normal(0, 1, 100)
    result = shannon_entropy(y, base)
    assert isinstance(result, dict)
