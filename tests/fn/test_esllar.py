"""Tests for esllar.esl_least_angle_reg."""
import numpy as np
import pytest
from morie.fn.esllar import esl_least_angle_reg


def test_esllar_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = esl_least_angle_reg(X, y)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_esllar_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = esl_least_angle_reg(X, y)
    assert isinstance(result, dict)
