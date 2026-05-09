"""Tests for agbtnm.alphazero_batch_norm."""
import numpy as np
import pytest
from moirais.fn.agbtnm import alphazero_batch_norm


def test_agbtnm_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    running_mean = np.random.default_rng(42).normal(0, 1, 100)
    running_var = np.random.default_rng(42).normal(0, 1, 100)
    momentum = np.random.default_rng(42).normal(0, 1, 100)
    result = alphazero_batch_norm(x, running_mean, running_var, momentum)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_agbtnm_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    running_mean = np.random.default_rng(42).normal(0, 1, 100)
    running_var = np.random.default_rng(42).normal(0, 1, 100)
    momentum = np.random.default_rng(42).normal(0, 1, 100)
    result = alphazero_batch_norm(x, running_mean, running_var, momentum)
    assert isinstance(result, dict)
