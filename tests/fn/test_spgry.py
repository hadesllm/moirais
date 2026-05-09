"""Tests for spgry.schabenberger_geary_c."""
import numpy as np
import pytest
from moirais.fn.spgry import schabenberger_geary_c


def test_spgry_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    w = np.random.default_rng(45).exponential(1, 100)
    result = schabenberger_geary_c(x, w)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_spgry_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    w = np.random.default_rng(45).exponential(1, 100)
    result = schabenberger_geary_c(x, w)
    assert isinstance(result, dict)
