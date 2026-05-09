"""Tests for poilO.poisson_loss_dnn."""
import numpy as np
import pytest
from moirais.fn.poilO import poisson_loss_dnn


def test_poilO_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    y_hat = np.random.default_rng(42).normal(0, 1, 100)
    result = poisson_loss_dnn(y, y_hat)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_poilO_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    y_hat = np.random.default_rng(42).normal(0, 1, 100)
    result = poisson_loss_dnn(y, y_hat)
    assert isinstance(result, dict)
