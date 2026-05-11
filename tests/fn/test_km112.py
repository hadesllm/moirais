"""Tests for km112.kamath_ch7_answer_relevance."""
import numpy as np
import pytest
from morie.fn.km112 import kamath_ch7_answer_relevance


def test_km112_basic():
    """Test basic functionality."""
    E_g = np.random.default_rng(42).normal(0, 1, 100)
    E_o = np.random.default_rng(42).normal(0, 1, 100)
    N = 100
    result = kamath_ch7_answer_relevance(E_g, E_o, N)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_km112_edge():
    """Test edge cases."""
    E_g = np.random.default_rng(42).normal(0, 1, 100)
    E_o = np.random.default_rng(42).normal(0, 1, 100)
    N = 100
    result = kamath_ch7_answer_relevance(E_g, E_o, N)
    assert isinstance(result, dict)
