"""Tests for hmohe.geron_one_hot_encoding."""
import numpy as np
import pytest
from moirais.fn.hmohe import geron_one_hot_encoding


def test_hmohe_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = geron_one_hot_encoding(X)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmohe_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = geron_one_hot_encoding(X)
    assert isinstance(result, dict)
