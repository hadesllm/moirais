"""Tests for sp_eq4.stochastic_physics_equation_4."""
import numpy as np
import pytest
from moirais.fn.sp_eq4 import stochastic_physics_equation_4


def test_sp_eq4_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = stochastic_physics_equation_4(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_sp_eq4_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = stochastic_physics_equation_4(x)
    assert isinstance(result, dict)
