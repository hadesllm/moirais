"""Tests for mkprod.mackinnon_distribution_products."""
import numpy as np
import pytest
from moirais.fn.mkprod import mackinnon_distribution_products


def test_mkprod_basic():
    """Test basic functionality."""
    a = np.random.default_rng(44).normal(0, 1, 100)
    b = np.random.default_rng(42).normal(0, 1, 100)
    se_a = np.random.default_rng(42).normal(0, 1, 100)
    se_b = np.random.default_rng(42).normal(0, 1, 100)
    result = mackinnon_distribution_products(a, b, se_a, se_b)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_mkprod_edge():
    """Test edge cases."""
    a = np.random.default_rng(44).normal(0, 1, 100)
    b = np.random.default_rng(42).normal(0, 1, 100)
    se_a = np.random.default_rng(42).normal(0, 1, 100)
    se_b = np.random.default_rng(42).normal(0, 1, 100)
    result = mackinnon_distribution_products(a, b, se_a, se_b)
    assert isinstance(result, dict)
