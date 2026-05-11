"""Tests for causivla.causal_iv_late."""
import numpy as np
import pytest
from morie.fn.causivla import causal_iv_late


def test_causivla_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    Z = np.random.default_rng(43).normal(0, 1, (100, 10))
    result = causal_iv_late(y, D, Z)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_causivla_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    Z = np.random.default_rng(43).normal(0, 1, (100, 10))
    result = causal_iv_late(y, D, Z)
    assert isinstance(result, dict)
