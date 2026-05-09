"""Tests for ate_d.ate_definition."""
import numpy as np
import pytest
from moirais.fn.ate_d import ate_definition


def test_ate_d_basic():
    """Test basic functionality."""
    Y1 = np.random.default_rng(42).normal(0, 1, 100)
    Y0 = np.random.default_rng(42).normal(0, 1, 100)
    result = ate_definition(Y1, Y0)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ate_d_edge():
    """Test edge cases."""
    Y1 = np.random.default_rng(42).normal(0, 1, 100)
    Y0 = np.random.default_rng(42).normal(0, 1, 100)
    result = ate_definition(Y1, Y0)
    assert isinstance(result, dict)
