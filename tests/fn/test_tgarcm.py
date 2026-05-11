"""Tests for tgarcm.tgarch_gjr."""
import numpy as np
import pytest
from morie.fn.tgarcm import tgarch_gjr


def test_tgarcm_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    p = 5
    q = np.random.default_rng(42).normal(0, 1, 100)
    result = tgarch_gjr(x, p, q)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_tgarcm_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    p = 5
    q = np.random.default_rng(42).normal(0, 1, 100)
    result = tgarch_gjr(x, p, q)
    assert isinstance(result, dict)
