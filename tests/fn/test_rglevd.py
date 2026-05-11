"""Tests for rglevd.rangayyan_levinson_durbin."""
import numpy as np
import pytest
from morie.fn.rglevd import rangayyan_levinson_durbin


def test_rglevd_basic():
    """Test basic functionality."""
    acf = np.random.default_rng(42).normal(0, 1, 100)
    order = 4
    result = rangayyan_levinson_durbin(acf, order)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rglevd_edge():
    """Test edge cases."""
    acf = np.random.default_rng(42).normal(0, 1, 100)
    order = 4
    result = rangayyan_levinson_durbin(acf, order)
    assert isinstance(result, dict)
