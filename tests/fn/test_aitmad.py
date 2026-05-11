"""Tests for aitmad.compositional_mad."""
import numpy as np
import pytest
from morie.fn.aitmad import compositional_mad


def test_aitmad_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = compositional_mad(X)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_aitmad_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = compositional_mad(X)
    assert isinstance(result, dict)
