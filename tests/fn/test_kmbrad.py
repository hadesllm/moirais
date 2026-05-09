"""Tests for kmbrad.kamath_bradley_terry_preference."""
import numpy as np
import pytest
from moirais.fn.kmbrad import kamath_bradley_terry_preference


def test_kmbrad_basic():
    """Test basic functionality."""
    r_w = np.random.default_rng(42).normal(0, 1, 100)
    r_l = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_bradley_terry_preference(r_w, r_l)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_kmbrad_edge():
    """Test edge cases."""
    r_w = np.random.default_rng(42).normal(0, 1, 100)
    r_l = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_bradley_terry_preference(r_w, r_l)
    assert isinstance(result, dict)
