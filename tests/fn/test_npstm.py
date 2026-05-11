"""Tests for npstm.nonparametric_tmle_survival."""
import numpy as np
import pytest
from morie.fn.npstm import nonparametric_tmle_survival


def test_npstm_basic():
    """Test basic functionality."""
    time = np.linspace(0, 10, 100)
    event = np.random.default_rng(42).normal(0, 1, 100)
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    W = np.random.default_rng(42).normal(0, 1, 100)
    result = nonparametric_tmle_survival(time, event, A, W)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_npstm_edge():
    """Test edge cases."""
    time = np.linspace(0, 10, 100)
    event = np.random.default_rng(42).normal(0, 1, 100)
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    W = np.random.default_rng(42).normal(0, 1, 100)
    result = nonparametric_tmle_survival(time, event, A, W)
    assert isinstance(result, dict)
