"""Tests for fzgn.fauzi_gn_edgeworth_correction."""
import numpy as np
import pytest
from moirais.fn.fzgn import fauzi_gn_edgeworth_correction


def test_fzgn_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    sigma_n = np.random.default_rng(42).normal(0, 1, 100)
    e_moments = np.random.default_rng(42).normal(0, 1, 100)
    bandwidth = 0.3
    result = fauzi_gn_edgeworth_correction(x, sigma_n, e_moments, bandwidth)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_fzgn_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    sigma_n = np.random.default_rng(42).normal(0, 1, 100)
    e_moments = np.random.default_rng(42).normal(0, 1, 100)
    bandwidth = 0.3
    result = fauzi_gn_edgeworth_correction(x, sigma_n, e_moments, bandwidth)
    assert isinstance(result, dict)
