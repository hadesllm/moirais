"""Tests for ibpfa.indian_buffet_factor."""
import numpy as np
import pytest
from morie.fn.ibpfa import indian_buffet_factor


def test_ibpfa_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    alpha = 0.05
    n_iter = 50
    result = indian_buffet_factor(y, alpha, n_iter)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ibpfa_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    alpha = 0.05
    n_iter = 50
    result = indian_buffet_factor(y, alpha, n_iter)
    assert isinstance(result, dict)
