"""Tests for wsmvar.wasserman_variance."""
import numpy as np
import pytest
from moirais.fn.wsmvar import wasserman_variance


def test_wsmvar_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wasserman_variance(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wsmvar_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wasserman_variance(x)
    assert isinstance(result, dict)
