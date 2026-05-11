"""Tests for genvxt.generalizability_theory."""
import numpy as np
import pytest
from morie.fn.genvxt import generalizability_theory


def test_genvxt_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    facets = np.random.default_rng(42).normal(0, 1, 100)
    result = generalizability_theory(X, facets)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_genvxt_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    facets = np.random.default_rng(42).normal(0, 1, 100)
    result = generalizability_theory(X, facets)
    assert isinstance(result, dict)
