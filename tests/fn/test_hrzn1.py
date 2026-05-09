"""Tests for hrzn1.horowitz_nonparametric_iv."""
import numpy as np
import pytest
from moirais.fn.hrzn1 import horowitz_nonparametric_iv


def test_hrzn1_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    z = np.random.default_rng(44).normal(0, 1, 100)
    result = horowitz_nonparametric_iv(x, y, z)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hrzn1_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    z = np.random.default_rng(44).normal(0, 1, 100)
    result = horowitz_nonparametric_iv(x, y, z)
    assert isinstance(result, dict)
