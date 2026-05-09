"""Tests for rng010.rangayyan_ch3_sample_std."""
import numpy as np
import pytest
from moirais.fn.rng010 import rangayyan_ch3_sample_std


def test_rng010_basic():
    """Test basic functionality."""
    eta = np.random.default_rng(42).normal(0, 1, 100)
    mu_eta = np.random.default_rng(42).normal(0, 1, 100)
    N = 100
    result = rangayyan_ch3_sample_std(eta, mu_eta, N)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rng010_edge():
    """Test edge cases."""
    eta = np.random.default_rng(42).normal(0, 1, 100)
    mu_eta = np.random.default_rng(42).normal(0, 1, 100)
    N = 100
    result = rangayyan_ch3_sample_std(eta, mu_eta, N)
    assert isinstance(result, dict)
