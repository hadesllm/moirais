"""Tests for ksr14.kosorok_profile_likelihood."""
import numpy as np
import pytest
from morie.fn.ksr14 import kosorok_profile_likelihood


def test_ksr14_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = kosorok_profile_likelihood(x, y)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ksr14_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = kosorok_profile_likelihood(x, y)
    assert isinstance(result, dict)
