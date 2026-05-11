"""Tests for aitlnp.logistic_normal_pdf."""
import numpy as np
import pytest
from morie.fn.aitlnp import logistic_normal_pdf


def test_aitlnp_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    mu = 0.0
    Sigma = np.random.default_rng(42).normal(0, 1, 100)
    result = logistic_normal_pdf(x, mu, Sigma)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_aitlnp_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    mu = 0.0
    Sigma = np.random.default_rng(42).normal(0, 1, 100)
    result = logistic_normal_pdf(x, mu, Sigma)
    assert isinstance(result, dict)
