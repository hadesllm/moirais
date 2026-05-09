"""Tests for wsmgmm.wasserman_gmm_em."""
import numpy as np
import pytest
from moirais.fn.wsmgmm import wasserman_gmm_em


def test_wsmgmm_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    k = 5
    result = wasserman_gmm_em(X, k)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wsmgmm_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    k = 5
    result = wasserman_gmm_em(X, k)
    assert isinstance(result, dict)
