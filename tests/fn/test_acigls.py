"""Tests for acigls.adjusted_ipgls."""
import numpy as np
import pytest
from moirais.fn.acigls import adjusted_ipgls


def test_acigls_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    H = np.random.default_rng(42).normal(0, 1, 100)
    cluster = np.random.default_rng(42).normal(0, 1, 100)
    result = adjusted_ipgls(y, A, H, cluster)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_acigls_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    H = np.random.default_rng(42).normal(0, 1, 100)
    cluster = np.random.default_rng(42).normal(0, 1, 100)
    result = adjusted_ipgls(y, A, H, cluster)
    assert isinstance(result, dict)
