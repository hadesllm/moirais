"""Tests for km032.kamath_ch2_seq2seq_loss."""
import numpy as np
import pytest
from moirais.fn.km032 import kamath_ch2_seq2seq_loss


def test_km032_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    xhat = np.random.default_rng(42).normal(0, 1, 100)
    i = np.random.default_rng(42).normal(0, 1, 100)
    j = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_ch2_seq2seq_loss(x, xhat, i, j)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_km032_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    xhat = np.random.default_rng(42).normal(0, 1, 100)
    i = np.random.default_rng(42).normal(0, 1, 100)
    j = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_ch2_seq2seq_loss(x, xhat, i, j)
    assert isinstance(result, dict)
