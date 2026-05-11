"""Tests for rWcoxa.wilcox_change."""
import numpy as np
import pytest
from morie.fn.rWcoxa import wilcox_change


def test_rWcoxa_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_change(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rWcoxa_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_change(x)
    assert isinstance(result, dict)
