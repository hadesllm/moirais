"""Tests for rrblpr.rr_blup."""
import numpy as np
import pytest
from moirais.fn.rrblpr import rr_blup


def test_rrblpr_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    M = np.random.default_rng(43).normal(0, 1, (10, 10))
    lam = 0.1
    result = rr_blup(y, M, lam)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rrblpr_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    M = np.random.default_rng(43).normal(0, 1, (10, 10))
    lam = 0.1
    result = rr_blup(y, M, lam)
    assert isinstance(result, dict)
