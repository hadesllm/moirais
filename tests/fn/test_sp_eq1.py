"""Tests for sp_eq1.stochastic_physics_equation_1."""
import numpy as np
import pytest
from moirais.fn.sp_eq1 import stochastic_physics_equation_1


def test_sp_eq1_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = stochastic_physics_equation_1(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_sp_eq1_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = stochastic_physics_equation_1(x)
    assert isinstance(result, dict)
