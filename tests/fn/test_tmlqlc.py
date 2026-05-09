"""Tests for tmlqlc.tmle_qlearning."""
import numpy as np
import pytest
from moirais.fn.tmlqlc import tmle_qlearning


def test_tmlqlc_basic():
    """Test basic functionality."""
    state = np.random.default_rng(42).normal(0, 1, 100)
    action = np.random.default_rng(42).normal(0, 1, 100)
    reward = np.random.default_rng(42).normal(0, 1, 100)
    time = np.linspace(0, 10, 100)
    result = tmle_qlearning(state, action, reward, time)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_tmlqlc_edge():
    """Test edge cases."""
    state = np.random.default_rng(42).normal(0, 1, 100)
    action = np.random.default_rng(42).normal(0, 1, 100)
    reward = np.random.default_rng(42).normal(0, 1, 100)
    time = np.linspace(0, 10, 100)
    result = tmle_qlearning(state, action, reward, time)
    assert isinstance(result, dict)
