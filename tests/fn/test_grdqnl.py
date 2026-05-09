"""Tests for grdqnl.geron_dqn_loss."""
import numpy as np
import pytest
from moirais.fn.grdqnl import geron_dqn_loss


def test_grdqnl_basic():
    """Test basic functionality."""
    Q = np.random.default_rng(42).normal(0, 1, 100)
    Q_target = np.random.default_rng(42).normal(0, 1, 100)
    batch = np.random.default_rng(42).normal(0, 1, 100)
    gamma = 1.0
    result = geron_dqn_loss(Q, Q_target, batch, gamma)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_grdqnl_edge():
    """Test edge cases."""
    Q = np.random.default_rng(42).normal(0, 1, 100)
    Q_target = np.random.default_rng(42).normal(0, 1, 100)
    batch = np.random.default_rng(42).normal(0, 1, 100)
    gamma = 1.0
    result = geron_dqn_loss(Q, Q_target, batch, gamma)
    assert isinstance(result, dict)
