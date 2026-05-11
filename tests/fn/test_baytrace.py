"""Tests for baytrace.trace_plot."""
import numpy as np
import pytest
from morie.fn.baytrace import trace_plot


def test_baytrace_basic():
    """Test basic functionality."""
    chains = np.random.default_rng(42).normal(0, 1, 100)
    result = trace_plot(chains)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_baytrace_edge():
    """Test edge cases."""
    chains = np.random.default_rng(42).normal(0, 1, 100)
    result = trace_plot(chains)
    assert isinstance(result, dict)
