"""Tests for km044.kamath_ch3_prompt_search_argmax."""
import numpy as np
import pytest
from morie.fn.km044 import kamath_ch3_prompt_search_argmax


def test_km044_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    z = np.random.default_rng(44).normal(0, 1, 100)
    theta = 0.0
    result = kamath_ch3_prompt_search_argmax(x, z, theta)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_km044_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    z = np.random.default_rng(44).normal(0, 1, 100)
    theta = 0.0
    result = kamath_ch3_prompt_search_argmax(x, z, theta)
    assert isinstance(result, dict)
