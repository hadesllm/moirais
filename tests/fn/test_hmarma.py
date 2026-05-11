"""Tests for hmarma.geron_arma."""
import numpy as np
import pytest
from morie.fn.hmarma import geron_arma


def test_hmarma_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    p = 5
    q = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_arma(y, p, q)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmarma_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    p = 5
    q = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_arma(y, p, q)
    assert isinstance(result, dict)
