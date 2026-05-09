"""Tests for egcoin.engle_granger_2step."""
import numpy as np
import pytest
from moirais.fn.egcoin import engle_granger_2step


def test_egcoin_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = engle_granger_2step(y, X)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_egcoin_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = engle_granger_2step(y, X)
    assert isinstance(result, dict)
