"""Tests for km117.kamath_ch8_bleu_final."""
import numpy as np
import pytest
from moirais.fn.km117 import kamath_ch8_bleu_final


def test_km117_basic():
    """Test basic functionality."""
    BP = np.random.default_rng(42).normal(0, 1, 100)
    p_n = np.random.default_rng(42).normal(0, 1, 100)
    N = 100
    result = kamath_ch8_bleu_final(BP, p_n, N)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_km117_edge():
    """Test edge cases."""
    BP = np.random.default_rng(42).normal(0, 1, 100)
    p_n = np.random.default_rng(42).normal(0, 1, 100)
    N = 100
    result = kamath_ch8_bleu_final(BP, p_n, N)
    assert isinstance(result, dict)
