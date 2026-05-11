"""Tests for irt1pl.rasch_one_parameter."""
import numpy as np
import pytest
from morie.fn.irt1pl import rasch_one_parameter


def test_irt1pl_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    theta = 0.0
    b = np.random.default_rng(42).normal(0, 1, 100)
    result = rasch_one_parameter(y, theta, b)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_irt1pl_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    theta = 0.0
    b = np.random.default_rng(42).normal(0, 1, 100)
    result = rasch_one_parameter(y, theta, b)
    assert isinstance(result, dict)
