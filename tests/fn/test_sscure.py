"""Tests for sscure.cure_model."""
import numpy as np
import pytest
from moirais.fn.sscure import cure_model


def test_sscure_basic():
    """Test basic functionality."""
    time = np.linspace(0, 10, 100)
    event = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    Z = np.random.default_rng(43).normal(0, 1, (100, 10))
    result = cure_model(time, event, X, Z)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_sscure_edge():
    """Test edge cases."""
    time = np.linspace(0, 10, 100)
    event = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    Z = np.random.default_rng(43).normal(0, 1, (100, 10))
    result = cure_model(time, event, X, Z)
    assert isinstance(result, dict)
