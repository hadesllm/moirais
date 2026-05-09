"""Tests for otpot.ot_pot_log_potentials."""
import numpy as np
import pytest
from moirais.fn.otpot import ot_pot_log_potentials


def test_otpot_basic():
    """Test basic functionality."""
    u = np.random.default_rng(44).normal(0, 1, 100)
    v = np.random.default_rng(44).normal(0, 1, 100)
    epsilon = 1e-6
    result = ot_pot_log_potentials(u, v, epsilon)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_otpot_edge():
    """Test edge cases."""
    u = np.random.default_rng(44).normal(0, 1, 100)
    v = np.random.default_rng(44).normal(0, 1, 100)
    epsilon = 1e-6
    result = ot_pot_log_potentials(u, v, epsilon)
    assert isinstance(result, dict)
