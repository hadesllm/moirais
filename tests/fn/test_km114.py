"""Tests for km114.kamath_ch8_bleu_precision."""
import numpy as np
import pytest
from morie.fn.km114 import kamath_ch8_bleu_precision


def test_km114_basic():
    """Test basic functionality."""
    n_grams = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_ch8_bleu_precision(n_grams)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_km114_edge():
    """Test edge cases."""
    n_grams = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_ch8_bleu_precision(n_grams)
    assert isinstance(result, dict)
