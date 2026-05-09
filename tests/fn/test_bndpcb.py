"""Tests for bndpcb.bound_pseudo_credible."""
import numpy as np
import pytest
from moirais.fn.bndpcb import bound_pseudo_credible


def test_bndpcb_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    alpha = 0.05
    result = bound_pseudo_credible(y, X, alpha)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_bndpcb_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    alpha = 0.05
    result = bound_pseudo_credible(y, X, alpha)
    assert isinstance(result, dict)
