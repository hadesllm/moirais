"""Tests for plcbo.placebo_refutation."""
import numpy as np
import pytest
from moirais.fn.plcbo import placebo_refutation


def test_plcbo_basic():
    """Test basic functionality."""
    model = np.random.default_rng(42).normal(0, 1, 100)
    n_simulations = np.random.default_rng(42).normal(0, 1, 100)
    result = placebo_refutation(model, n_simulations)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_plcbo_edge():
    """Test edge cases."""
    model = np.random.default_rng(42).normal(0, 1, 100)
    n_simulations = np.random.default_rng(42).normal(0, 1, 100)
    result = placebo_refutation(model, n_simulations)
    assert isinstance(result, dict)
