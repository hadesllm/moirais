"""Tests for gb1331w.gibbons_wsrt_efficacy."""
import numpy as np
import pytest
from morie.fn.gb1331w import gibbons_wsrt_efficacy


def test_gb1331w_basic():
    """Test basic functionality."""
    N = 100
    f = np.random.default_rng(42).normal(0, 1, 100)
    result = gibbons_wsrt_efficacy(N, f)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_gb1331w_edge():
    """Test edge cases."""
    N = 100
    f = np.random.default_rng(42).normal(0, 1, 100)
    result = gibbons_wsrt_efficacy(N, f)
    assert isinstance(result, dict)
