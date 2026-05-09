"""Tests for rng041.rangayyan_ch3_lsi_series_intermediate."""
import numpy as np
import pytest
from moirais.fn.rng041 import rangayyan_ch3_lsi_series_intermediate


def test_rng041_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    h_1 = np.random.default_rng(42).normal(0, 1, 100)
    n = 100
    result = rangayyan_ch3_lsi_series_intermediate(x, h_1, n)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rng041_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    h_1 = np.random.default_rng(42).normal(0, 1, 100)
    n = 100
    result = rangayyan_ch3_lsi_series_intermediate(x, h_1, n)
    assert isinstance(result, dict)
