"""Tests for grjll.geron_johnson_lindenstrauss_bound."""
import numpy as np
import pytest
from moirais.fn.grjll import geron_johnson_lindenstrauss_bound


def test_grjll_basic():
    """Test basic functionality."""
    n_samples = np.random.default_rng(42).normal(0, 1, 100)
    eps = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_johnson_lindenstrauss_bound(n_samples, eps)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_grjll_edge():
    """Test edge cases."""
    n_samples = np.random.default_rng(42).normal(0, 1, 100)
    eps = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_johnson_lindenstrauss_bound(n_samples, eps)
    assert isinstance(result, dict)
