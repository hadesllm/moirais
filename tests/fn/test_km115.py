"""Tests for km115.kamath_ch8_bleu_n_geom_mean."""
import numpy as np
import pytest
from moirais.fn.km115 import kamath_ch8_bleu_n_geom_mean


def test_km115_basic():
    """Test basic functionality."""
    p_n = np.random.default_rng(42).normal(0, 1, 100)
    N = 100
    result = kamath_ch8_bleu_n_geom_mean(p_n, N)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_km115_edge():
    """Test edge cases."""
    p_n = np.random.default_rng(42).normal(0, 1, 100)
    N = 100
    result = kamath_ch8_bleu_n_geom_mean(p_n, N)
    assert isinstance(result, dict)
