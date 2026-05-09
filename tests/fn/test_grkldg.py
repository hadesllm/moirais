"""Tests for grkldg.geron_kl_divergence_gaussian."""
import numpy as np
import pytest
from moirais.fn.grkldg import geron_kl_divergence_gaussian


def test_grkldg_basic():
    """Test basic functionality."""
    mu = 0.0
    logvar = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_kl_divergence_gaussian(mu, logvar)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_grkldg_edge():
    """Test edge cases."""
    mu = 0.0
    logvar = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_kl_divergence_gaussian(mu, logvar)
    assert isinstance(result, dict)
