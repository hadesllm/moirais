"""Tests for hrzphvnp.horowitz_ph_frailty_nonpar."""
import numpy as np
import pytest
from moirais.fn.hrzphvnp import horowitz_ph_frailty_nonpar


def test_hrzphvnp_basic():
    """Test basic functionality."""
    t = np.linspace(0, 10, 100)
    x = np.random.default_rng(42).normal(0, 1, 100)
    event = np.random.default_rng(42).normal(0, 1, 100)
    result = horowitz_ph_frailty_nonpar(t, x, event)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hrzphvnp_edge():
    """Test edge cases."""
    t = np.linspace(0, 10, 100)
    x = np.random.default_rng(42).normal(0, 1, 100)
    event = np.random.default_rng(42).normal(0, 1, 100)
    result = horowitz_ph_frailty_nonpar(t, x, event)
    assert isinstance(result, dict)
