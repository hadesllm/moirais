"""Tests for otwsg.ot_wasserstein_gauss."""
import numpy as np
import pytest
from moirais.fn.otwsg import ot_wasserstein_gauss


def test_otwsg_basic():
    """Test basic functionality."""
    mu1 = np.random.default_rng(42).normal(0, 1, 100)
    Sigma1 = np.random.default_rng(42).normal(0, 1, 100)
    mu2 = np.random.default_rng(42).normal(0, 1, 100)
    Sigma2 = np.random.default_rng(42).normal(0, 1, 100)
    result = ot_wasserstein_gauss(mu1, Sigma1, mu2, Sigma2)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_otwsg_edge():
    """Test edge cases."""
    mu1 = np.random.default_rng(42).normal(0, 1, 100)
    Sigma1 = np.random.default_rng(42).normal(0, 1, 100)
    mu2 = np.random.default_rng(42).normal(0, 1, 100)
    Sigma2 = np.random.default_rng(42).normal(0, 1, 100)
    result = ot_wasserstein_gauss(mu1, Sigma1, mu2, Sigma2)
    assert isinstance(result, dict)
