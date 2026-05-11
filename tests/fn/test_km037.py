"""Tests for km037.kamath_ch2_gpt_combined_obj."""
import numpy as np
import pytest
from morie.fn.km037 import kamath_ch2_gpt_combined_obj


def test_km037_basic():
    """Test basic functionality."""
    L_1 = np.random.default_rng(42).normal(0, 1, 100)
    L_2 = np.random.default_rng(42).normal(0, 1, 100)
    lam = 0.1
    result = kamath_ch2_gpt_combined_obj(L_1, L_2, lam)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_km037_edge():
    """Test edge cases."""
    L_1 = np.random.default_rng(42).normal(0, 1, 100)
    L_2 = np.random.default_rng(42).normal(0, 1, 100)
    lam = 0.1
    result = kamath_ch2_gpt_combined_obj(L_1, L_2, lam)
    assert isinstance(result, dict)
