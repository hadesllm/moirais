"""Tests for otsobm.ot_sobolev_w1."""
import numpy as np
import pytest
from moirais.fn.otsobm import ot_sobolev_w1


def test_otsobm_basic():
    """Test basic functionality."""
    mu = 0.0
    nu = np.random.default_rng(42).normal(0, 1, 100)
    Laplace_inv = np.random.default_rng(42).normal(0, 1, 100)
    result = ot_sobolev_w1(mu, nu, Laplace_inv)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_otsobm_edge():
    """Test edge cases."""
    mu = 0.0
    nu = np.random.default_rng(42).normal(0, 1, 100)
    Laplace_inv = np.random.default_rng(42).normal(0, 1, 100)
    result = ot_sobolev_w1(mu, nu, Laplace_inv)
    assert isinstance(result, dict)
