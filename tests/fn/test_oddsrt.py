"""Tests for oddsrt.odds_ratio."""
import numpy as np
import pytest
from moirais.fn.oddsrt import odds_ratio


def test_oddsrt_basic():
    """Test basic functionality."""
    a = np.random.default_rng(44).normal(0, 1, 100)
    b = np.random.default_rng(42).normal(0, 1, 100)
    c = np.random.default_rng(42).normal(0, 1, 100)
    d = 5
    result = odds_ratio(a, b, c, d)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_oddsrt_edge():
    """Test edge cases."""
    a = np.random.default_rng(44).normal(0, 1, 100)
    b = np.random.default_rng(42).normal(0, 1, 100)
    c = np.random.default_rng(42).normal(0, 1, 100)
    d = 5
    result = odds_ratio(a, b, c, d)
    assert isinstance(result, dict)
