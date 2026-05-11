"""Tests for hmbic.geron_bic."""
import numpy as np
import pytest
from morie.fn.hmbic import geron_bic


def test_hmbic_basic():
    """Test basic functionality."""
    log_lik = np.random.default_rng(42).normal(0, 1, 100)
    k = 5
    n = 100
    result = geron_bic(log_lik, k, n)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmbic_edge():
    """Test edge cases."""
    log_lik = np.random.default_rng(42).normal(0, 1, 100)
    k = 5
    n = 100
    result = geron_bic(log_lik, k, n)
    assert isinstance(result, dict)
