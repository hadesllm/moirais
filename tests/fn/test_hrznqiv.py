"""Tests for hrznqiv.horowitz_nonpar_quantile_iv."""
import numpy as np
import pytest
from moirais.fn.hrznqiv import horowitz_nonpar_quantile_iv


def test_hrznqiv_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    w = np.random.default_rng(45).exponential(1, 100)
    tau = 0.1
    result = horowitz_nonpar_quantile_iv(x, y, w, tau)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hrznqiv_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    w = np.random.default_rng(45).exponential(1, 100)
    tau = 0.1
    result = horowitz_nonpar_quantile_iv(x, y, w, tau)
    assert isinstance(result, dict)
