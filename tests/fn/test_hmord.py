"""Tests for hmord.geron_ordinal_encoding."""
import numpy as np
import pytest
from morie.fn.hmord import geron_ordinal_encoding


def test_hmord_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    categories = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_ordinal_encoding(X, categories)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmord_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    categories = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_ordinal_encoding(X, categories)
    assert isinstance(result, dict)
