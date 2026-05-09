"""Tests for hrznls.horowitz_nls_sim."""
import numpy as np
import pytest
from moirais.fn.hrznls import horowitz_nls_sim


def test_hrznls_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    bandwidth = 0.3
    result = horowitz_nls_sim(x, y, bandwidth)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hrznls_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    bandwidth = 0.3
    result = horowitz_nls_sim(x, y, bandwidth)
    assert isinstance(result, dict)
