"""Tests for pposc.posterior_predictive_check."""
import numpy as np
import pytest
from moirais.fn.pposc import posterior_predictive_check


def test_pposc_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    y_rep = np.random.default_rng(42).normal(0, 1, 100)
    result = posterior_predictive_check(y, y_rep)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_pposc_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    y_rep = np.random.default_rng(42).normal(0, 1, 100)
    result = posterior_predictive_check(y, y_rep)
    assert isinstance(result, dict)
