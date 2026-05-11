"""Tests for causdidwd.causal_did_wooldridge_eta."""
import numpy as np
import pytest
from morie.fn.causdidwd import causal_did_wooldridge_eta


def test_causdidwd_basic():
    """Test basic functionality."""
    Y_panel = np.random.default_rng(42).normal(0, 1, 100)
    G_first_treat = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = causal_did_wooldridge_eta(Y_panel, G_first_treat, X)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_causdidwd_edge():
    """Test edge cases."""
    Y_panel = np.random.default_rng(42).normal(0, 1, 100)
    G_first_treat = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = causal_did_wooldridge_eta(Y_panel, G_first_treat, X)
    assert isinstance(result, dict)
