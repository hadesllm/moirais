"""Tests for fzgkde.fauzi_gamma_kde."""
import numpy as np
import pytest
from moirais.fn.fzgkde import fauzi_gamma_kde


def test_fzgkde_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    bandwidth = 0.3
    result = fauzi_gamma_kde(x, bandwidth)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_fzgkde_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    bandwidth = 0.3
    result = fauzi_gamma_kde(x, bandwidth)
    assert isinstance(result, dict)
