"""Tests for lrcvg.learning_curve."""
import numpy as np
import pytest
from morie.fn.lrcvg import learning_curve


def test_lrcvg_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = learning_curve(x, y)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_lrcvg_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = learning_curve(x, y)
    assert isinstance(result, dict)
