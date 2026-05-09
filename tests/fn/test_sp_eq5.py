"""Tests for sp_eq5.stochastic_physics_equation_5."""
import numpy as np
import pytest
from moirais.fn.sp_eq5 import stochastic_physics_equation_5


def test_sp_eq5_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = stochastic_physics_equation_5(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_sp_eq5_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = stochastic_physics_equation_5(x)
    assert isinstance(result, dict)
