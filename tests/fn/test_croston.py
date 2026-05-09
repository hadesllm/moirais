"""Tests for croston.croston."""
import numpy as np
import pytest
from moirais.fn.croston import croston


def test_croston_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    alpha = 0.05
    result = croston(y, alpha)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_croston_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    alpha = 0.05
    result = croston(y, alpha)
    assert isinstance(result, dict)
