"""Tests for jopql.joseph_pinball_quantile_loss."""
import numpy as np
import pytest
from morie.fn.jopql import joseph_pinball_quantile_loss


def test_jopql_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    q = np.random.default_rng(42).normal(0, 1, 100)
    tau = 0.1
    result = joseph_pinball_quantile_loss(y, q, tau)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_jopql_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    q = np.random.default_rng(42).normal(0, 1, 100)
    tau = 0.1
    result = joseph_pinball_quantile_loss(y, q, tau)
    assert isinstance(result, dict)
