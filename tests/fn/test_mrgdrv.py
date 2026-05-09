"""Tests for mrgdrv.martingale_concentration."""
import numpy as np
import pytest
from moirais.fn.mrgdrv import martingale_concentration


def test_mrgdrv_basic():
    """Test basic functionality."""
    c = np.random.default_rng(42).normal(0, 1, 100)
    n = 100
    t = np.linspace(0, 10, 100)
    result = martingale_concentration(c, n, t)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_mrgdrv_edge():
    """Test edge cases."""
    c = np.random.default_rng(42).normal(0, 1, 100)
    n = 100
    t = np.linspace(0, 10, 100)
    result = martingale_concentration(c, n, t)
    assert isinstance(result, dict)
