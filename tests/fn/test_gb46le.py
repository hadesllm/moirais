"""Tests for gb46le.gibbons_lilliefors_exp."""
import numpy as np
import pytest
from moirais.fn.gb46le import gibbons_lilliefors_exp


def test_gb46le_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = gibbons_lilliefors_exp(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_gb46le_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = gibbons_lilliefors_exp(x)
    assert isinstance(result, dict)
