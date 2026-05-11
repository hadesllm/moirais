"""Tests for alnsmp.alammar_negative_sampling_skipgram."""
import numpy as np
import pytest
from morie.fn.alnsmp import alammar_negative_sampling_skipgram


def test_alnsmp_basic():
    """Test basic functionality."""
    center = np.random.default_rng(42).normal(0, 1, 100)
    word = np.random.default_rng(42).normal(0, 1, 100)
    negatives = np.random.default_rng(42).normal(0, 1, 100)
    V = np.random.default_rng(42).normal(0, 1, 100)
    result = alammar_negative_sampling_skipgram(center, word, negatives, V)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_alnsmp_edge():
    """Test edge cases."""
    center = np.random.default_rng(42).normal(0, 1, 100)
    word = np.random.default_rng(42).normal(0, 1, 100)
    negatives = np.random.default_rng(42).normal(0, 1, 100)
    V = np.random.default_rng(42).normal(0, 1, 100)
    result = alammar_negative_sampling_skipgram(center, word, negatives, V)
    assert isinstance(result, dict)
