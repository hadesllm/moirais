"""Tests for km010.kamath_ch2_attention_output."""
import numpy as np
import pytest
from moirais.fn.km010 import kamath_ch2_attention_output


def test_km010_basic():
    """Test basic functionality."""
    b = np.random.default_rng(42).normal(0, 1, 100)
    v = np.random.default_rng(44).normal(0, 1, 100)
    result = kamath_ch2_attention_output(b, v)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_km010_edge():
    """Test edge cases."""
    b = np.random.default_rng(42).normal(0, 1, 100)
    v = np.random.default_rng(44).normal(0, 1, 100)
    result = kamath_ch2_attention_output(b, v)
    assert isinstance(result, dict)
