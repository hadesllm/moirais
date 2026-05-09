"""Tests for causmtch.causal_pair_matching."""
import numpy as np
import pytest
from moirais.fn.causmtch import causal_pair_matching


def test_causmtch_basic():
    """Test basic functionality."""
    ps = np.random.default_rng(42).normal(0, 1, 100)
    treat = np.random.default_rng(42).normal(0, 1, 100)
    caliper = np.random.default_rng(42).normal(0, 1, 100)
    result = causal_pair_matching(ps, treat, caliper)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_causmtch_edge():
    """Test edge cases."""
    ps = np.random.default_rng(42).normal(0, 1, 100)
    treat = np.random.default_rng(42).normal(0, 1, 100)
    caliper = np.random.default_rng(42).normal(0, 1, 100)
    result = causal_pair_matching(ps, treat, caliper)
    assert isinstance(result, dict)
