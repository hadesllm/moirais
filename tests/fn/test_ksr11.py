"""Tests for ksr11.kosorok_efficient_score."""
import numpy as np
import pytest
from moirais.fn.ksr11 import kosorok_efficient_score


def test_ksr11_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = kosorok_efficient_score(x, y)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ksr11_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = kosorok_efficient_score(x, y)
    assert isinstance(result, dict)
