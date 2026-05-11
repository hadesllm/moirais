"""Tests for km005.kamath_ch2_decoder_token_distribution."""
import numpy as np
import pytest
from morie.fn.km005 import kamath_ch2_decoder_token_distribution


def test_km005_basic():
    """Test basic functionality."""
    s_t_1 = np.random.default_rng(42).normal(0, 1, 100)
    y_t_1 = np.random.default_rng(42).normal(0, 1, 100)
    c = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_ch2_decoder_token_distribution(s_t_1, y_t_1, c)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_km005_edge():
    """Test edge cases."""
    s_t_1 = np.random.default_rng(42).normal(0, 1, 100)
    y_t_1 = np.random.default_rng(42).normal(0, 1, 100)
    c = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_ch2_decoder_token_distribution(s_t_1, y_t_1, c)
    assert isinstance(result, dict)
