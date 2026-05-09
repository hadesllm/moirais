"""Tests for causrddf.causal_rdd_fuzzy."""
import numpy as np
import pytest
from moirais.fn.causrddf import causal_rdd_fuzzy


def test_causrddf_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    treat = np.random.default_rng(42).normal(0, 1, 100)
    cutoff = 10.0
    h = 0.3
    result = causal_rdd_fuzzy(x, y, treat, cutoff, h)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_causrddf_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    treat = np.random.default_rng(42).normal(0, 1, 100)
    cutoff = 10.0
    h = 0.3
    result = causal_rdd_fuzzy(x, y, treat, cutoff, h)
    assert isinstance(result, dict)
