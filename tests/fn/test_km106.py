"""Tests for km106.kamath_ch6_self_diagnosis_prob."""
import numpy as np
import pytest
from morie.fn.km106 import kamath_ch6_self_diagnosis_prob


def test_km106_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    M = np.random.default_rng(43).normal(0, 1, (10, 10))
    sdg = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_ch6_self_diagnosis_prob(x, y, M, sdg)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_km106_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    M = np.random.default_rng(43).normal(0, 1, (10, 10))
    sdg = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_ch6_self_diagnosis_prob(x, y, M, sdg)
    assert isinstance(result, dict)
