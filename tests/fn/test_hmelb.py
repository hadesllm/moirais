"""Tests for hmelb.geron_elbo."""
import numpy as np
import pytest
from morie.fn.hmelb import geron_elbo


def test_hmelb_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    mu = 0.0
    log_sigma = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_elbo(x, mu, log_sigma)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmelb_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    mu = 0.0
    log_sigma = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_elbo(x, mu, log_sigma)
    assert isinstance(result, dict)
