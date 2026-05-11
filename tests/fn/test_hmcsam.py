"""Tests for hmcsam.hamiltonian_mc."""
import numpy as np
import pytest
from morie.fn.hmcsam import hamiltonian_mc


def test_hmcsam_basic():
    """Test basic functionality."""
    log_p = np.random.default_rng(42).normal(0, 1, 100)
    grad_log_p = np.random.default_rng(42).normal(0, 1, 100)
    x0 = np.random.default_rng(42).normal(0, 1, 100)
    step_size = 100
    L = np.random.default_rng(42).normal(0, 1, 100)
    result = hamiltonian_mc(log_p, grad_log_p, x0, step_size, L)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmcsam_edge():
    """Test edge cases."""
    log_p = np.random.default_rng(42).normal(0, 1, 100)
    grad_log_p = np.random.default_rng(42).normal(0, 1, 100)
    x0 = np.random.default_rng(42).normal(0, 1, 100)
    step_size = 100
    L = np.random.default_rng(42).normal(0, 1, 100)
    result = hamiltonian_mc(log_p, grad_log_p, x0, step_size, L)
    assert isinstance(result, dict)
