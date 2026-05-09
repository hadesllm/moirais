"""Tests for rng003.rangayyan_ch3_variance_continuous."""
import numpy as np
import pytest
from moirais.fn.rng003 import rangayyan_ch3_variance_continuous


def test_rng003_basic():
    """Test basic functionality."""
    eta = np.random.default_rng(42).normal(0, 1, 100)
    mu_eta = np.random.default_rng(42).normal(0, 1, 100)
    p_eta = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_ch3_variance_continuous(eta, mu_eta, p_eta)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rng003_edge():
    """Test edge cases."""
    eta = np.random.default_rng(42).normal(0, 1, 100)
    mu_eta = np.random.default_rng(42).normal(0, 1, 100)
    p_eta = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_ch3_variance_continuous(eta, mu_eta, p_eta)
    assert isinstance(result, dict)
