"""Tests for hmxln.geron_xlnet."""
import numpy as np
import pytest
from morie.fn.hmxln import geron_xlnet


def test_hmxln_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    n_layers = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_xlnet(X, n_layers)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmxln_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    n_layers = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_xlnet(X, n_layers)
    assert isinstance(result, dict)
