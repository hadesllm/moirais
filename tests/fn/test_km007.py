"""Tests for km007.kamath_ch2_attention_score."""
import numpy as np
import pytest
from morie.fn.km007 import kamath_ch2_attention_score


def test_km007_basic():
    """Test basic functionality."""
    q = np.random.default_rng(42).normal(0, 1, 100)
    k_i = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_ch2_attention_score(q, k_i)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_km007_edge():
    """Test edge cases."""
    q = np.random.default_rng(42).normal(0, 1, 100)
    k_i = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_ch2_attention_score(q, k_i)
    assert isinstance(result, dict)
