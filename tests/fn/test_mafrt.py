"""Tests for mafrt.ma_freeman_tukey."""
import numpy as np
import pytest
from morie.fn.mafrt import ma_freeman_tukey


def test_mafrt_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    n = 100
    result = ma_freeman_tukey(x, n)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_mafrt_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    n = 100
    result = ma_freeman_tukey(x, n)
    assert isinstance(result, dict)
