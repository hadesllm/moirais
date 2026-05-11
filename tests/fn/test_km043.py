"""Tests for km043.kamath_ch3_prompt_softmax_label."""
import numpy as np
import pytest
from morie.fn.km043 import kamath_ch3_prompt_softmax_label


def test_km043_basic():
    """Test basic functionality."""
    w = np.random.default_rng(45).exponential(1, 100)
    h_z = np.random.default_rng(42).normal(0, 1, 100)
    M = np.random.default_rng(43).normal(0, 1, (10, 10))
    result = kamath_ch3_prompt_softmax_label(w, h_z, M)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_km043_edge():
    """Test edge cases."""
    w = np.random.default_rng(45).exponential(1, 100)
    h_z = np.random.default_rng(42).normal(0, 1, 100)
    M = np.random.default_rng(43).normal(0, 1, (10, 10))
    result = kamath_ch3_prompt_softmax_label(w, h_z, M)
    assert isinstance(result, dict)
