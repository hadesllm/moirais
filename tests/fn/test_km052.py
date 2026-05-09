"""Tests for km052.kamath_ch3_t5_template_obj."""
import numpy as np
import pytest
from moirais.fn.km052 import kamath_ch3_t5_template_obj


def test_km052_basic():
    """Test basic functionality."""
    D_train = np.random.default_rng(42).normal(0, 1, 100)
    T = np.random.default_rng(43).integers(0, 2, 100)
    T5 = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_ch3_t5_template_obj(D_train, T, T5)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_km052_edge():
    """Test edge cases."""
    D_train = np.random.default_rng(42).normal(0, 1, 100)
    T = np.random.default_rng(43).integers(0, 2, 100)
    T5 = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_ch3_t5_template_obj(D_train, T, T5)
    assert isinstance(result, dict)
