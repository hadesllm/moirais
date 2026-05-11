"""Tests for causdidcs.causal_did_callaway_sa."""
import numpy as np
import pytest
from morie.fn.causdidcs import causal_did_callaway_sa


def test_causdidcs_basic():
    """Test basic functionality."""
    Y_panel = np.random.default_rng(42).normal(0, 1, 100)
    G_first_treat = np.random.default_rng(42).normal(0, 1, 100)
    X_baseline = np.random.default_rng(42).normal(0, 1, 100)
    result = causal_did_callaway_sa(Y_panel, G_first_treat, X_baseline)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_causdidcs_edge():
    """Test edge cases."""
    Y_panel = np.random.default_rng(42).normal(0, 1, 100)
    G_first_treat = np.random.default_rng(42).normal(0, 1, 100)
    X_baseline = np.random.default_rng(42).normal(0, 1, 100)
    result = causal_did_callaway_sa(Y_panel, G_first_treat, X_baseline)
    assert isinstance(result, dict)
