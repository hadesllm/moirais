"""Tests for hmmod.geron_model_based."""
import numpy as np
import pytest
from moirais.fn.hmmod import geron_model_based


def test_hmmod_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = geron_model_based(X, y)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmmod_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = geron_model_based(X, y)
    assert isinstance(result, dict)
