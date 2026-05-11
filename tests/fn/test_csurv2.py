"""Tests for csurv2.causal_survival_blp."""
import numpy as np
import pytest
from morie.fn.csurv2 import causal_survival_blp


def test_csurv2_basic():
    """Test basic functionality."""
    time = np.linspace(0, 10, 100)
    event = np.random.default_rng(42).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = causal_survival_blp(time, event, D, X)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_csurv2_edge():
    """Test edge cases."""
    time = np.linspace(0, 10, 100)
    event = np.random.default_rng(42).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = causal_survival_blp(time, event, D, X)
    assert isinstance(result, dict)
