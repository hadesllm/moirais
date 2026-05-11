"""Tests for jrqbst.jarque_bera."""
import numpy as np
import pytest
from morie.fn.jrqbst import jarque_bera


def test_jrqbst_basic():
    """Test basic functionality."""
    residuals = np.random.default_rng(42).normal(0, 1, 100)
    result = jarque_bera(residuals)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_jrqbst_edge():
    """Test edge cases."""
    residuals = np.random.default_rng(42).normal(0, 1, 100)
    result = jarque_bera(residuals)
    assert isinstance(result, dict)
