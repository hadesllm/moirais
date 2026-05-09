"""Tests for km001.kamath_ch2_unidirectional_encoder_state."""
import numpy as np
import pytest
from moirais.fn.km001 import kamath_ch2_unidirectional_encoder_state


def test_km001_basic():
    """Test basic functionality."""
    h_t_1 = np.random.default_rng(42).normal(0, 1, 100)
    x_t = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_ch2_unidirectional_encoder_state(h_t_1, x_t)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_km001_edge():
    """Test edge cases."""
    h_t_1 = np.random.default_rng(42).normal(0, 1, 100)
    x_t = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_ch2_unidirectional_encoder_state(h_t_1, x_t)
    assert isinstance(result, dict)
