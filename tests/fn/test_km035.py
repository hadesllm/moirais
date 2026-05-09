"""Tests for km035.kamath_ch2_gpt_supervised_softmax."""
import numpy as np
import pytest
from moirais.fn.km035 import kamath_ch2_gpt_supervised_softmax


def test_km035_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    h = 0.3
    W_y = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_ch2_gpt_supervised_softmax(x, h, W_y)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_km035_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    h = 0.3
    W_y = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_ch2_gpt_supervised_softmax(x, h, W_y)
    assert isinstance(result, dict)
