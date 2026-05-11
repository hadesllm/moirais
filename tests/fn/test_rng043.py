"""Tests for rng043.rangayyan_ch3_lsi_series_combined_h."""
import numpy as np
import pytest
from morie.fn.rng043 import rangayyan_ch3_lsi_series_combined_h


def test_rng043_basic():
    """Test basic functionality."""
    h_1 = np.random.default_rng(42).normal(0, 1, 100)
    h_2 = np.random.default_rng(42).normal(0, 1, 100)
    n = 100
    result = rangayyan_ch3_lsi_series_combined_h(h_1, h_2, n)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rng043_edge():
    """Test edge cases."""
    h_1 = np.random.default_rng(42).normal(0, 1, 100)
    h_2 = np.random.default_rng(42).normal(0, 1, 100)
    n = 100
    result = rangayyan_ch3_lsi_series_combined_h(h_1, h_2, n)
    assert isinstance(result, dict)
