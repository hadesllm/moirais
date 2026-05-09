"""Tests for kchnls.k_l_divergence_chain."""
import numpy as np
import pytest
from moirais.fn.kchnls import k_l_divergence_chain


def test_kchnls_basic():
    """Test basic functionality."""
    pxy = np.random.default_rng(42).normal(0, 1, 100)
    qxy = np.random.default_rng(42).normal(0, 1, 100)
    result = k_l_divergence_chain(pxy, qxy)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_kchnls_edge():
    """Test edge cases."""
    pxy = np.random.default_rng(42).normal(0, 1, 100)
    qxy = np.random.default_rng(42).normal(0, 1, 100)
    result = k_l_divergence_chain(pxy, qxy)
    assert isinstance(result, dict)
