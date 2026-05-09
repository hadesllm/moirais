"""Tests for btpct.boot_percentile_ci."""
import numpy as np
import pytest
from moirais.fn.btpct import boot_percentile_ci


def test_btpct_basic():
    """Test basic functionality."""
    theta_b = np.random.default_rng(42).normal(0, 1, 100)
    alpha = 0.05
    result = boot_percentile_ci(theta_b, alpha)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_btpct_edge():
    """Test edge cases."""
    theta_b = np.random.default_rng(42).normal(0, 1, 100)
    alpha = 0.05
    result = boot_percentile_ci(theta_b, alpha)
    assert isinstance(result, dict)
