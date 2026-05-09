"""Tests for wsmmcd.wasserman_mcdiarmid."""
import numpy as np
import pytest
from moirais.fn.wsmmcd import wasserman_mcdiarmid


def test_wsmmcd_basic():
    """Test basic functionality."""
    t = np.linspace(0, 10, 100)
    c = np.random.default_rng(42).normal(0, 1, 100)
    result = wasserman_mcdiarmid(t, c)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wsmmcd_edge():
    """Test edge cases."""
    t = np.linspace(0, 10, 100)
    c = np.random.default_rng(42).normal(0, 1, 100)
    result = wasserman_mcdiarmid(t, c)
    assert isinstance(result, dict)
