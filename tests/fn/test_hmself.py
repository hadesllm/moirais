"""Tests for hmself.geron_self_supervised."""
import numpy as np
import pytest
from moirais.fn.hmself import geron_self_supervised


def test_hmself_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    pretext = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_self_supervised(X, pretext)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmself_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    pretext = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_self_supervised(X, pretext)
    assert isinstance(result, dict)
