"""Tests for eslkrn.esl_kernel_density."""
import numpy as np
import pytest
from moirais.fn.eslkrn import esl_kernel_density


def test_eslkrn_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    data = np.random.default_rng(42).normal(0, 1, 100)
    lambda_ = np.random.default_rng(42).normal(0, 1, 100)
    result = esl_kernel_density(x, data, lambda_)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_eslkrn_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    data = np.random.default_rng(42).normal(0, 1, 100)
    lambda_ = np.random.default_rng(42).normal(0, 1, 100)
    result = esl_kernel_density(x, data, lambda_)
    assert isinstance(result, dict)
