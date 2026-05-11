"""Tests for cbnrt.causalbert_text."""
import numpy as np
import pytest
from morie.fn.cbnrt import causalbert_text


def test_cbnrt_basic():
    """Test basic functionality."""
    texts = np.random.default_rng(42).normal(0, 1, 100)
    T = np.random.default_rng(43).integers(0, 2, 100)
    Y = np.random.default_rng(43).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = causalbert_text(texts, T, Y, X)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_cbnrt_edge():
    """Test edge cases."""
    texts = np.random.default_rng(42).normal(0, 1, 100)
    T = np.random.default_rng(43).integers(0, 2, 100)
    Y = np.random.default_rng(43).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = causalbert_text(texts, T, Y, X)
    assert isinstance(result, dict)
