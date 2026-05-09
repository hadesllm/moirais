"""Tests for causscg.causal_generalised_sc."""
import numpy as np
import pytest
from moirais.fn.causscg import causal_generalised_sc


def test_causscg_basic():
    """Test basic functionality."""
    Y_panel = np.random.default_rng(42).normal(0, 1, 100)
    treated_idx = np.random.default_rng(42).normal(0, 1, 100)
    treat_time = np.random.default_rng(42).normal(0, 1, 100)
    r = 10
    result = causal_generalised_sc(Y_panel, treated_idx, treat_time, r)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_causscg_edge():
    """Test edge cases."""
    Y_panel = np.random.default_rng(42).normal(0, 1, 100)
    treated_idx = np.random.default_rng(42).normal(0, 1, 100)
    treat_time = np.random.default_rng(42).normal(0, 1, 100)
    r = 10
    result = causal_generalised_sc(Y_panel, treated_idx, treat_time, r)
    assert isinstance(result, dict)
