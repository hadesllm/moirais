"""Tests for km004.kamath_ch2_decoder_hidden_state."""
import numpy as np
import pytest
from morie.fn.km004 import kamath_ch2_decoder_hidden_state


def test_km004_basic():
    """Test basic functionality."""
    s_t_1 = np.random.default_rng(42).normal(0, 1, 100)
    y_t_1 = np.random.default_rng(42).normal(0, 1, 100)
    c = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_ch2_decoder_hidden_state(s_t_1, y_t_1, c)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_km004_edge():
    """Test edge cases."""
    s_t_1 = np.random.default_rng(42).normal(0, 1, 100)
    y_t_1 = np.random.default_rng(42).normal(0, 1, 100)
    c = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_ch2_decoder_hidden_state(s_t_1, y_t_1, c)
    assert isinstance(result, dict)
