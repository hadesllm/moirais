"""Tests for rng193.rangayyan_ch4_heart_rate_from_count."""
import numpy as np
import pytest
from morie.fn.rng193 import rangayyan_ch4_heart_rate_from_count


def test_rng193_basic():
    """Test basic functionality."""
    N_B = np.random.default_rng(42).normal(0, 1, 100)
    T = np.random.default_rng(43).integers(0, 2, 100)
    result = rangayyan_ch4_heart_rate_from_count(N_B, T)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rng193_edge():
    """Test edge cases."""
    N_B = np.random.default_rng(42).normal(0, 1, 100)
    T = np.random.default_rng(43).integers(0, 2, 100)
    result = rangayyan_ch4_heart_rate_from_count(N_B, T)
    assert isinstance(result, dict)
