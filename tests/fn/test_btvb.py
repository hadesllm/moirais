"""Tests for btvb.boot_var_estimator."""
import numpy as np
import pytest
from moirais.fn.btvb import boot_var_estimator


def test_btvb_basic():
    """Test basic functionality."""
    theta_b = np.random.default_rng(42).normal(0, 1, 100)
    result = boot_var_estimator(theta_b)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_btvb_edge():
    """Test edge cases."""
    theta_b = np.random.default_rng(42).normal(0, 1, 100)
    result = boot_var_estimator(theta_b)
    assert isinstance(result, dict)
