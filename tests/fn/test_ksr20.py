"""Tests for ksr20.kosorok_censoring_survival."""
import numpy as np
import pytest
from morie.fn.ksr20 import kosorok_censoring_survival


def test_ksr20_basic():
    """Test basic functionality."""
    t = np.linspace(0, 10, 100)
    event = np.random.default_rng(42).normal(0, 1, 100)
    result = kosorok_censoring_survival(t, event)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ksr20_edge():
    """Test edge cases."""
    t = np.linspace(0, 10, 100)
    event = np.random.default_rng(42).normal(0, 1, 100)
    result = kosorok_censoring_survival(t, event)
    assert isinstance(result, dict)
