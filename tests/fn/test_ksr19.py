"""Tests for ksr19.kosorok_cox_partial_likelihood."""
import numpy as np
import pytest
from moirais.fn.ksr19 import kosorok_cox_partial_likelihood


def test_ksr19_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    t = np.linspace(0, 10, 100)
    event = np.random.default_rng(42).normal(0, 1, 100)
    result = kosorok_cox_partial_likelihood(x, t, event)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ksr19_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    t = np.linspace(0, 10, 100)
    event = np.random.default_rng(42).normal(0, 1, 100)
    result = kosorok_cox_partial_likelihood(x, t, event)
    assert isinstance(result, dict)
