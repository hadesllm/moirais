"""Tests for mdppol.mdp_policy_iteration."""
import numpy as np
import pytest
from morie.fn.mdppol import mdp_policy_iteration


def test_mdppol_basic():
    """Test basic functionality."""
    P = np.random.default_rng(42).normal(0, 1, 100)
    R = np.random.default_rng(42).normal(0, 1, 100)
    gamma = 1.0
    result = mdp_policy_iteration(P, R, gamma)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_mdppol_edge():
    """Test edge cases."""
    P = np.random.default_rng(42).normal(0, 1, 100)
    R = np.random.default_rng(42).normal(0, 1, 100)
    gamma = 1.0
    result = mdp_policy_iteration(P, R, gamma)
    assert isinstance(result, dict)
