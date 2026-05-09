"""Tests for deciA.deci_model."""
import numpy as np
import pytest
from moirais.fn.deciA import deci_model


def test_deciA_basic():
    """Test basic functionality."""
    data = np.random.default_rng(42).normal(0, 1, 100)
    n_samples = np.random.default_rng(42).normal(0, 1, 100)
    lr = np.random.default_rng(42).normal(0, 1, 100)
    result = deci_model(data, n_samples, lr)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_deciA_edge():
    """Test edge cases."""
    data = np.random.default_rng(42).normal(0, 1, 100)
    n_samples = np.random.default_rng(42).normal(0, 1, 100)
    lr = np.random.default_rng(42).normal(0, 1, 100)
    result = deci_model(data, n_samples, lr)
    assert isinstance(result, dict)
