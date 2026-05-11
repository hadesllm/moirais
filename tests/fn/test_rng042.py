"""Tests for rng042.rangayyan_ch3_lsi_series_total."""
import numpy as np
import pytest
from morie.fn.rng042 import rangayyan_ch3_lsi_series_total


def test_rng042_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    h_1 = np.random.default_rng(42).normal(0, 1, 100)
    h_2 = np.random.default_rng(42).normal(0, 1, 100)
    n = 100
    result = rangayyan_ch3_lsi_series_total(x, h_1, h_2, n)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rng042_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    h_1 = np.random.default_rng(42).normal(0, 1, 100)
    h_2 = np.random.default_rng(42).normal(0, 1, 100)
    n = 100
    result = rangayyan_ch3_lsi_series_total(x, h_1, h_2, n)
    assert isinstance(result, dict)
