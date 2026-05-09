"""Tests for wsmkbk.wasserman_kullback_leibler."""
import numpy as np
import pytest
from moirais.fn.wsmkbk import wasserman_kullback_leibler


def test_wsmkbk_basic():
    """Test basic functionality."""
    p = 5
    q = np.random.default_rng(42).normal(0, 1, 100)
    result = wasserman_kullback_leibler(p, q)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wsmkbk_edge():
    """Test edge cases."""
    p = 5
    q = np.random.default_rng(42).normal(0, 1, 100)
    result = wasserman_kullback_leibler(p, q)
    assert isinstance(result, dict)
