"""Tests for causovlap.causal_overlap_diagnostic."""
import numpy as np
import pytest
from moirais.fn.causovlap import causal_overlap_diagnostic


def test_causovlap_basic():
    """Test basic functionality."""
    ps = np.random.default_rng(42).normal(0, 1, 100)
    treat = np.random.default_rng(42).normal(0, 1, 100)
    result = causal_overlap_diagnostic(ps, treat)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_causovlap_edge():
    """Test edge cases."""
    ps = np.random.default_rng(42).normal(0, 1, 100)
    treat = np.random.default_rng(42).normal(0, 1, 100)
    result = causal_overlap_diagnostic(ps, treat)
    assert isinstance(result, dict)
