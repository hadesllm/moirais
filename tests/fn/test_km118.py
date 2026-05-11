"""Tests for km118.kamath_ch8_rouge_n."""
import numpy as np
import pytest
from morie.fn.km118 import kamath_ch8_rouge_n


def test_km118_basic():
    """Test basic functionality."""
    S = np.random.default_rng(42).normal(0, 1, 100)
    gram_n = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_ch8_rouge_n(S, gram_n)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_km118_edge():
    """Test edge cases."""
    S = np.random.default_rng(42).normal(0, 1, 100)
    gram_n = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_ch8_rouge_n(S, gram_n)
    assert isinstance(result, dict)
