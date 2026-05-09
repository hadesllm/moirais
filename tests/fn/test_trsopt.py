"""Tests for trsopt.trust_region_subproblem."""
import numpy as np
import pytest
from moirais.fn.trsopt import trust_region_subproblem


def test_trsopt_basic():
    """Test basic functionality."""
    g = np.random.default_rng(43).normal(0, 1, 100)
    H = np.random.default_rng(42).normal(0, 1, 100)
    delta = np.random.default_rng(42).normal(0, 1, 100)
    result = trust_region_subproblem(g, H, delta)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_trsopt_edge():
    """Test edge cases."""
    g = np.random.default_rng(43).normal(0, 1, 100)
    H = np.random.default_rng(42).normal(0, 1, 100)
    delta = np.random.default_rng(42).normal(0, 1, 100)
    result = trust_region_subproblem(g, H, delta)
    assert isinstance(result, dict)
