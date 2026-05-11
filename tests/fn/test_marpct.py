"""Tests for marpct.ma_percent_heterogeneity_R2."""
import numpy as np
import pytest
from morie.fn.marpct import ma_percent_heterogeneity_R2


def test_marpct_basic():
    """Test basic functionality."""
    tau2_full = np.random.default_rng(42).normal(0, 1, 100)
    tau2_null = np.random.default_rng(42).normal(0, 1, 100)
    result = ma_percent_heterogeneity_R2(tau2_full, tau2_null)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_marpct_edge():
    """Test edge cases."""
    tau2_full = np.random.default_rng(42).normal(0, 1, 100)
    tau2_null = np.random.default_rng(42).normal(0, 1, 100)
    result = ma_percent_heterogeneity_R2(tau2_full, tau2_null)
    assert isinstance(result, dict)
