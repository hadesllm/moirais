"""Tests for km008.kamath_ch2_attention_softmax_weights."""
import numpy as np
import pytest
from moirais.fn.km008 import kamath_ch2_attention_softmax_weights


def test_km008_basic():
    """Test basic functionality."""
    a = np.random.default_rng(44).normal(0, 1, 100)
    result = kamath_ch2_attention_softmax_weights(a)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_km008_edge():
    """Test edge cases."""
    a = np.random.default_rng(44).normal(0, 1, 100)
    result = kamath_ch2_attention_softmax_weights(a)
    assert isinstance(result, dict)
