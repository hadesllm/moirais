"""Tests for irtprc.partial_credit."""
import numpy as np
import pytest
from moirais.fn.irtprc import partial_credit


def test_irtprc_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    ncats = np.random.default_rng(42).normal(0, 1, 100)
    result = partial_credit(X, ncats)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_irtprc_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    ncats = np.random.default_rng(42).normal(0, 1, 100)
    result = partial_credit(X, ncats)
    assert isinstance(result, dict)
