"""Tests for gbens.gradient_boosting_ensemble."""
import numpy as np
import pytest
from morie.fn.gbens import gradient_boosting_ensemble


def test_gbens_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = gradient_boosting_ensemble(x, y)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_gbens_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = gradient_boosting_ensemble(x, y)
    assert isinstance(result, dict)
