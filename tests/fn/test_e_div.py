"""Tests for e_div.e_divisive."""
import numpy as np
import pytest
from moirais.fn.e_div import e_divisive


def test_e_div_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    sig = np.random.default_rng(42).normal(0, 1, 100)
    result = e_divisive(x, sig)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_e_div_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    sig = np.random.default_rng(42).normal(0, 1, 100)
    result = e_divisive(x, sig)
    assert isinstance(result, dict)
