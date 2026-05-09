"""Tests for flowmm.max_flow_min_cut."""
import numpy as np
import pytest
from moirais.fn.flowmm import max_flow_min_cut


def test_flowmm_basic():
    """Test basic functionality."""
    G = np.eye(10)
    source = np.random.default_rng(42).normal(0, 1, 100)
    sink = np.random.default_rng(42).normal(0, 1, 100)
    result = max_flow_min_cut(G, source, sink)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_flowmm_edge():
    """Test edge cases."""
    G = np.eye(10)
    source = np.random.default_rng(42).normal(0, 1, 100)
    sink = np.random.default_rng(42).normal(0, 1, 100)
    result = max_flow_min_cut(G, source, sink)
    assert isinstance(result, dict)
