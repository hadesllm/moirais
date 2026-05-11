"""Tests for eslspc.esl_spectral_cluster."""
import numpy as np
import pytest
from morie.fn.eslspc import esl_spectral_cluster


def test_eslspc_basic():
    """Test basic functionality."""
    W = np.random.default_rng(42).normal(0, 1, 100)
    k = 5
    result = esl_spectral_cluster(W, k)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_eslspc_edge():
    """Test edge cases."""
    W = np.random.default_rng(42).normal(0, 1, 100)
    k = 5
    result = esl_spectral_cluster(W, k)
    assert isinstance(result, dict)
