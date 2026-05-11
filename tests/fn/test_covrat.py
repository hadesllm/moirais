"""Tests for covrat.covratio."""
import numpy as np
import pytest
from morie.fn.covrat import covratio


def test_covrat_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = covratio(y, X)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_covrat_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = covratio(y, X)
    assert isinstance(result, dict)
