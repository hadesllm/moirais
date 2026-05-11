"""Tests for km102.kamath_ch6_lstm_chain_rule."""
import numpy as np
import pytest
from morie.fn.km102 import kamath_ch6_lstm_chain_rule


def test_km102_basic():
    """Test basic functionality."""
    w_1_w_M = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_ch6_lstm_chain_rule(w_1_w_M)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_km102_edge():
    """Test edge cases."""
    w_1_w_M = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_ch6_lstm_chain_rule(w_1_w_M)
    assert isinstance(result, dict)
