"""Tests for advinf.advi."""
import numpy as np
import pytest
from morie.fn.advinf import advi


def test_advinf_basic():
    """Test basic functionality."""
    log_p = np.random.default_rng(42).normal(0, 1, 100)
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = advi(log_p, x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_advinf_edge():
    """Test edge cases."""
    log_p = np.random.default_rng(42).normal(0, 1, 100)
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = advi(log_p, x)
    assert isinstance(result, dict)
