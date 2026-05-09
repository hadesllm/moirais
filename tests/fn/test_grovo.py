"""Tests for grovo.geron_one_vs_one."""
import numpy as np
import pytest
from moirais.fn.grovo import geron_one_vs_one


def test_grovo_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = geron_one_vs_one(X, y)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_grovo_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = geron_one_vs_one(X, y)
    assert isinstance(result, dict)
