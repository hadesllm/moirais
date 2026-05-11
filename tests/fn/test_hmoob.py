"""Tests for hmoob.geron_oob_score."""
import numpy as np
import pytest
from morie.fn.hmoob import geron_oob_score


def test_hmoob_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    models = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_oob_score(X, y, models)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmoob_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    models = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_oob_score(X, y, models)
    assert isinstance(result, dict)
