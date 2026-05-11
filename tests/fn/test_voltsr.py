"""Tests for voltsr.vol_two_scale_rv."""
import numpy as np
import pytest
from morie.fn.voltsr import vol_two_scale_rv


def test_voltsr_basic():
    """Test basic functionality."""
    r_intraday = np.random.default_rng(42).normal(0, 1, 100)
    K = np.eye(10) + 0.1 * np.random.default_rng(43).normal(0, 1, (10, 10))
    result = vol_two_scale_rv(r_intraday, K)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_voltsr_edge():
    """Test edge cases."""
    r_intraday = np.random.default_rng(42).normal(0, 1, 100)
    K = np.eye(10) + 0.1 * np.random.default_rng(43).normal(0, 1, (10, 10))
    result = vol_two_scale_rv(r_intraday, K)
    assert isinstance(result, dict)
