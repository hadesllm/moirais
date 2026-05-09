"""Tests for rpgad.rdp_to_eps_delta."""
import numpy as np
import pytest
from moirais.fn.rpgad import rdp_to_eps_delta


def test_rpgad_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    alpha = 0.05
    epsilon_R = np.random.default_rng(42).normal(0, 1, 100)
    delta = np.random.default_rng(42).normal(0, 1, 100)
    result = rdp_to_eps_delta(y, alpha, epsilon_R, delta)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rpgad_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    alpha = 0.05
    epsilon_R = np.random.default_rng(42).normal(0, 1, 100)
    delta = np.random.default_rng(42).normal(0, 1, 100)
    result = rdp_to_eps_delta(y, alpha, epsilon_R, delta)
    assert isinstance(result, dict)
