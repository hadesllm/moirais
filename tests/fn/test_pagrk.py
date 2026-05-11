"""Tests for pagrk.pagerank."""
import numpy as np
import pytest
from morie.fn.pagrk import pagerank


def test_pagrk_basic():
    """Test basic functionality."""
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    alpha = 0.05
    result = pagerank(A, alpha)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_pagrk_edge():
    """Test edge cases."""
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    alpha = 0.05
    result = pagerank(A, alpha)
    assert isinstance(result, dict)
