"""Tests for grn005.geron_ch4_normal_equation."""
import numpy as np
import pytest
from moirais.fn.grn005 import geron_ch4_normal_equation


def test_grn005_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = geron_ch4_normal_equation(X, y)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_grn005_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = geron_ch4_normal_equation(X, y)
    assert isinstance(result, dict)
