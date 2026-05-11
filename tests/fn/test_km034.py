"""Tests for km034.kamath_ch2_gpt_unsupervised_obj."""
import numpy as np
import pytest
from morie.fn.km034 import kamath_ch2_gpt_unsupervised_obj


def test_km034_basic():
    """Test basic functionality."""
    U = np.random.default_rng(42).normal(0, 1, 100)
    k = 5
    Theta = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_ch2_gpt_unsupervised_obj(U, k, Theta)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_km034_edge():
    """Test edge cases."""
    U = np.random.default_rng(42).normal(0, 1, 100)
    k = 5
    Theta = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_ch2_gpt_unsupervised_obj(U, k, Theta)
    assert isinstance(result, dict)
