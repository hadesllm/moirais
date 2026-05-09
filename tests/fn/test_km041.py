"""Tests for km041.kamath_ch2_mixtral_swiglu_moe."""
import numpy as np
import pytest
from moirais.fn.km041 import kamath_ch2_mixtral_swiglu_moe


def test_km041_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    W_g = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_ch2_mixtral_swiglu_moe(x, W_g)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_km041_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    W_g = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_ch2_mixtral_swiglu_moe(x, W_g)
    assert isinstance(result, dict)
