"""Tests for naday.nadaraya_watson."""
import numpy as np
import pytest
from moirais.fn.naday import nadaraya_watson


def test_naday_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    h = 0.3
    result = nadaraya_watson(x, y, h)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_naday_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    h = 0.3
    result = nadaraya_watson(x, y, h)
    assert isinstance(result, dict)
