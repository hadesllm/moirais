"""Tests for ppsamp.pps_sampling."""
import numpy as np
import pytest
from moirais.fn.ppsamp import pps_sampling


def test_ppsamp_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    size = np.random.default_rng(42).normal(0, 1, 100)
    n = 100
    result = pps_sampling(y, size, n)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ppsamp_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    size = np.random.default_rng(42).normal(0, 1, 100)
    n = 100
    result = pps_sampling(y, size, n)
    assert isinstance(result, dict)
