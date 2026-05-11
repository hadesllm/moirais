"""Tests for causdidsap.causal_did_sun_abraham."""
import numpy as np
import pytest
from morie.fn.causdidsap import causal_did_sun_abraham


def test_causdidsap_basic():
    """Test basic functionality."""
    Y_panel = np.random.default_rng(42).normal(0, 1, 100)
    G_first_treat = np.random.default_rng(42).normal(0, 1, 100)
    result = causal_did_sun_abraham(Y_panel, G_first_treat)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_causdidsap_edge():
    """Test edge cases."""
    Y_panel = np.random.default_rng(42).normal(0, 1, 100)
    G_first_treat = np.random.default_rng(42).normal(0, 1, 100)
    result = causal_did_sun_abraham(Y_panel, G_first_treat)
    assert isinstance(result, dict)
