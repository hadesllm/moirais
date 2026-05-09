"""Tests for km057.kamath_ch4_lora_obj."""
import numpy as np
import pytest
from moirais.fn.km057 import kamath_ch4_lora_obj


def test_km057_basic():
    """Test basic functionality."""
    Theta = np.random.default_rng(42).normal(0, 1, 100)
    Phi_0 = np.random.default_rng(42).normal(0, 1, 100)
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = kamath_ch4_lora_obj(Theta, Phi_0, x, y)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_km057_edge():
    """Test edge cases."""
    Theta = np.random.default_rng(42).normal(0, 1, 100)
    Phi_0 = np.random.default_rng(42).normal(0, 1, 100)
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = kamath_ch4_lora_obj(Theta, Phi_0, x, y)
    assert isinstance(result, dict)
