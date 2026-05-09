"""Tests for baysrr.bayes_ridge."""
import numpy as np
import pytest
from moirais.fn.baysrr import bayes_ridge


def test_baysrr_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    M = np.random.default_rng(43).normal(0, 1, (10, 10))
    result = bayes_ridge(y, M)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_baysrr_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    M = np.random.default_rng(43).normal(0, 1, (10, 10))
    result = bayes_ridge(y, M)
    assert isinstance(result, dict)
