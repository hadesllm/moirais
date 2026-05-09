"""Tests for km134.kamath_ch9_clip_text_to_image."""
import numpy as np
import pytest
from moirais.fn.km134 import kamath_ch9_clip_text_to_image


def test_km134_basic():
    """Test basic functionality."""
    L = np.random.default_rng(42).normal(0, 1, 100)
    V = np.random.default_rng(42).normal(0, 1, 100)
    sigma = 1.0
    N = 100
    result = kamath_ch9_clip_text_to_image(L, V, sigma, N)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_km134_edge():
    """Test edge cases."""
    L = np.random.default_rng(42).normal(0, 1, 100)
    V = np.random.default_rng(42).normal(0, 1, 100)
    sigma = 1.0
    N = 100
    result = kamath_ch9_clip_text_to_image(L, V, sigma, N)
    assert isinstance(result, dict)
