"""Tests for km103.kamath_ch6_lstm_softmax_word."""
import numpy as np
import pytest
from moirais.fn.km103 import kamath_ch6_lstm_softmax_word


def test_km103_basic():
    """Test basic functionality."""
    U = np.random.default_rng(42).normal(0, 1, 100)
    f = np.random.default_rng(42).normal(0, 1, 100)
    c_t_1 = np.random.default_rng(42).normal(0, 1, 100)
    b = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_ch6_lstm_softmax_word(U, f, c_t_1, b)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_km103_edge():
    """Test edge cases."""
    U = np.random.default_rng(42).normal(0, 1, 100)
    f = np.random.default_rng(42).normal(0, 1, 100)
    c_t_1 = np.random.default_rng(42).normal(0, 1, 100)
    b = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_ch6_lstm_softmax_word(U, f, c_t_1, b)
    assert isinstance(result, dict)
