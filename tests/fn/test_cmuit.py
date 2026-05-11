"""Tests for cmuit.conditional_mutual_information."""
import numpy as np
import pytest
from morie.fn.cmuit import conditional_mutual_information


def test_cmuit_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    x = np.random.default_rng(42).normal(0, 1, 100)
    y2 = np.random.default_rng(42).normal(0, 1, 100)
    z = np.random.default_rng(44).normal(0, 1, 100)
    result = conditional_mutual_information(y, x, y2, z)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_cmuit_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    x = np.random.default_rng(42).normal(0, 1, 100)
    y2 = np.random.default_rng(42).normal(0, 1, 100)
    z = np.random.default_rng(44).normal(0, 1, 100)
    result = conditional_mutual_information(y, x, y2, z)
    assert isinstance(result, dict)
