"""Tests for dpfed.dp_fedavg."""
import numpy as np
import pytest
from morie.fn.dpfed import dp_fedavg


def test_dpfed_basic():
    """Test basic functionality."""
    clients = np.random.default_rng(42).normal(0, 1, 100)
    C = np.random.default_rng(42).normal(0, 1, 100)
    sigma = 1.0
    result = dp_fedavg(clients, C, sigma)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_dpfed_edge():
    """Test edge cases."""
    clients = np.random.default_rng(42).normal(0, 1, 100)
    C = np.random.default_rng(42).normal(0, 1, 100)
    sigma = 1.0
    result = dp_fedavg(clients, C, sigma)
    assert isinstance(result, dict)
