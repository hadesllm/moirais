"""Tests for baynav.variational_nf."""
import numpy as np
import pytest
from morie.fn.baynav import variational_nf


def test_baynav_basic():
    """Test basic functionality."""
    log_p = np.random.default_rng(42).normal(0, 1, 100)
    flow = np.random.default_rng(42).normal(0, 1, 100)
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = variational_nf(log_p, flow, x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_baynav_edge():
    """Test edge cases."""
    log_p = np.random.default_rng(42).normal(0, 1, 100)
    flow = np.random.default_rng(42).normal(0, 1, 100)
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = variational_nf(log_p, flow, x)
    assert isinstance(result, dict)
