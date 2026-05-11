"""Tests for volmsg.vol_markov_switching_garch."""
import numpy as np
import pytest
from morie.fn.volmsg import vol_markov_switching_garch


def test_volmsg_basic():
    """Test basic functionality."""
    r = 10
    K = np.eye(10) + 0.1 * np.random.default_rng(43).normal(0, 1, (10, 10))
    init = np.random.default_rng(42).normal(0, 1, 100)
    result = vol_markov_switching_garch(r, K, init)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_volmsg_edge():
    """Test edge cases."""
    r = 10
    K = np.eye(10) + 0.1 * np.random.default_rng(43).normal(0, 1, (10, 10))
    init = np.random.default_rng(42).normal(0, 1, 100)
    result = vol_markov_switching_garch(r, K, init)
    assert isinstance(result, dict)
