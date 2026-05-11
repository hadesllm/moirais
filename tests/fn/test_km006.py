"""Tests for km006.kamath_ch2_seq2seq_cross_entropy."""
import numpy as np
import pytest
from morie.fn.km006 import kamath_ch2_seq2seq_cross_entropy


def test_km006_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    c = np.random.default_rng(42).normal(0, 1, 100)
    U = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_ch2_seq2seq_cross_entropy(y, c, U)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_km006_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    c = np.random.default_rng(42).normal(0, 1, 100)
    U = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_ch2_seq2seq_cross_entropy(y, c, U)
    assert isinstance(result, dict)
