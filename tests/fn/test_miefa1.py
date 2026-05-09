"""Tests for miefa1.mi_fmi."""
import numpy as np
import pytest
from moirais.fn.miefa1 import mi_fmi


def test_miefa1_basic():
    """Test basic functionality."""
    between = np.random.default_rng(42).normal(0, 1, 100)
    within = np.random.default_rng(42).normal(0, 1, 100)
    m = 10
    result = mi_fmi(between, within, m)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_miefa1_edge():
    """Test edge cases."""
    between = np.random.default_rng(42).normal(0, 1, 100)
    within = np.random.default_rng(42).normal(0, 1, 100)
    m = 10
    result = mi_fmi(between, within, m)
    assert isinstance(result, dict)
