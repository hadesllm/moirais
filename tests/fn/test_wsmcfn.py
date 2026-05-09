"""Tests for wsmcfn.wasserman_char_fn."""
import numpy as np
import pytest
from moirais.fn.wsmcfn import wasserman_char_fn


def test_wsmcfn_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    t = np.linspace(0, 10, 100)
    result = wasserman_char_fn(x, t)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wsmcfn_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    t = np.linspace(0, 10, 100)
    result = wasserman_char_fn(x, t)
    assert isinstance(result, dict)
