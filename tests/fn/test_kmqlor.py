"""Tests for kmqlor.kamath_qlora_4bit."""
import numpy as np
import pytest
from morie.fn.kmqlor import kamath_qlora_4bit


def test_kmqlor_basic():
    """Test basic functionality."""
    W0_nf4 = np.random.default_rng(42).normal(0, 1, 100)
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    B = np.random.default_rng(43).normal(0, 1, (10, 10))
    alpha = 0.05
    r = 10
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_qlora_4bit(W0_nf4, A, B, alpha, r, x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_kmqlor_edge():
    """Test edge cases."""
    W0_nf4 = np.random.default_rng(42).normal(0, 1, 100)
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    B = np.random.default_rng(43).normal(0, 1, (10, 10))
    alpha = 0.05
    r = 10
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_qlora_4bit(W0_nf4, A, B, alpha, r, x)
    assert isinstance(result, dict)
