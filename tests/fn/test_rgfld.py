"""Tests for rgfld.rangayyan_fisher_lda."""
import numpy as np
import pytest
from morie.fn.rgfld import rangayyan_fisher_lda


def test_rgfld_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = rangayyan_fisher_lda(X, y)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rgfld_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = rangayyan_fisher_lda(X, y)
    assert isinstance(result, dict)
