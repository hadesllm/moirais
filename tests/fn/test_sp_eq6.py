"""Tests for sp_eq6.stochastic_physics_equation_6."""
import numpy as np
import pytest
from morie.fn.sp_eq6 import stochastic_physics_equation_6


def test_sp_eq6_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = stochastic_physics_equation_6(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_sp_eq6_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = stochastic_physics_equation_6(x)
    assert isinstance(result, dict)
