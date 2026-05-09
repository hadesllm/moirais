"""Tests for sp_s6u17.stochastic_physics_section_6_unnumbered_17."""
import numpy as np
import pytest
from moirais.fn.sp_s6u17 import stochastic_physics_section_6_unnumbered_17


def test_sp_s6u17_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = stochastic_physics_section_6_unnumbered_17(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_sp_s6u17_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = stochastic_physics_section_6_unnumbered_17(x)
    assert isinstance(result, dict)
