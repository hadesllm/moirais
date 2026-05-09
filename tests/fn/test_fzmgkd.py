"""Tests for fzmgkd.fauzi_modified_gamma_kde."""
import numpy as np
import pytest
from moirais.fn.fzmgkd import fauzi_modified_gamma_kde


def test_fzmgkd_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    bandwidth = 0.3
    a = np.random.default_rng(44).normal(0, 1, 100)
    result = fauzi_modified_gamma_kde(x, bandwidth, a)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_fzmgkd_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    bandwidth = 0.3
    a = np.random.default_rng(44).normal(0, 1, 100)
    result = fauzi_modified_gamma_kde(x, bandwidth, a)
    assert isinstance(result, dict)
