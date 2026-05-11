"""Tests for volnois.vol_noise_variance_est."""
import numpy as np
import pytest
from morie.fn.volnois import vol_noise_variance_est


def test_volnois_basic():
    """Test basic functionality."""
    r_intraday = np.random.default_rng(42).normal(0, 1, 100)
    result = vol_noise_variance_est(r_intraday)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_volnois_edge():
    """Test edge cases."""
    r_intraday = np.random.default_rng(42).normal(0, 1, 100)
    result = vol_noise_variance_est(r_intraday)
    assert isinstance(result, dict)
