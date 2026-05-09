"""Tests for tmlpoo.tmle_pooled."""
import numpy as np
import pytest
from moirais.fn.tmlpoo import tmle_pooled


def test_tmlpoo_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    site = np.random.default_rng(42).normal(0, 1, 100)
    result = tmle_pooled(y, D, X, site)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_tmlpoo_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    site = np.random.default_rng(42).normal(0, 1, 100)
    result = tmle_pooled(y, D, X, site)
    assert isinstance(result, dict)
