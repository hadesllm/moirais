"""Tests for aitzlk.compositional_zero_lrda."""
import numpy as np
import pytest
from moirais.fn.aitzlk import compositional_zero_lrda


def test_aitzlk_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    n_iter = 50
    result = compositional_zero_lrda(X, n_iter)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_aitzlk_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    n_iter = 50
    result = compositional_zero_lrda(X, n_iter)
    assert isinstance(result, dict)
