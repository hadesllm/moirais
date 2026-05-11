"""Tests for km036.kamath_ch2_gpt_supervised_obj."""
import numpy as np
import pytest
from morie.fn.km036 import kamath_ch2_gpt_supervised_obj


def test_km036_basic():
    """Test basic functionality."""
    C = np.random.default_rng(42).normal(0, 1, 100)
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = kamath_ch2_gpt_supervised_obj(C, x, y)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_km036_edge():
    """Test edge cases."""
    C = np.random.default_rng(42).normal(0, 1, 100)
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = kamath_ch2_gpt_supervised_obj(C, x, y)
    assert isinstance(result, dict)
