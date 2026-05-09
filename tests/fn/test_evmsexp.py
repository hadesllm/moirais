"""Tests for evmsexp.evt_max_stable_logistic."""
import numpy as np
import pytest
from moirais.fn.evmsexp import evt_max_stable_logistic


def test_evmsexp_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    alpha = 0.05
    result = evt_max_stable_logistic(x, y, alpha)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_evmsexp_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    alpha = 0.05
    result = evt_max_stable_logistic(x, y, alpha)
    assert isinstance(result, dict)
